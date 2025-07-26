use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{read_dir, read_to_string, write};
use std::process::Command;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time;
use wsgg::Connection;

/// AI Chat Bot
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Location of the file containing the cookie for the bot to use
    #[arg(short, long)]
    cookie: String,

    /// Use the dev environement (chat2.strims.gg)
    #[arg(short, long, default_value_t = false)]
    dev: bool,

    /// API endpoint host and port (e.g., localhost:8080)
    #[arg(long, default_value = "localhost:8080")]
    api_host: String,
}

#[derive(Serialize, Deserialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OllamaResponse {
    response: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct WeatherResponse {
    current_condition: Vec<CurrentCondition>,
}

#[derive(Serialize, Deserialize, Debug)]
struct CurrentCondition {
    #[serde(rename = "FeelsLikeC")]
    feels_like_c: String,
    #[serde(rename = "FeelsLikeF")]
    feels_like_f: String,
    cloudcover: String,
    humidity: String,
    #[serde(rename = "localObsDateTime")]
    local_obs_date_time: String,
    observation_time: String,
    #[serde(rename = "precipInches")]
    precip_inches: String,
    #[serde(rename = "precipMM")]
    precip_mm: String,
    pressure: String,
    #[serde(rename = "pressureInches")]
    pressure_inches: String,
    #[serde(rename = "temp_C")]
    temp_c: String,
    #[serde(rename = "temp_F")]
    temp_f: String,
    #[serde(rename = "uvIndex")]
    uv_index: String,
    visibility: String,
    #[serde(rename = "visibilityMiles")]
    visibility_miles: String,
    #[serde(rename = "weatherCode")]
    weather_code: String,
    #[serde(rename = "weatherDesc")]
    weather_desc: Vec<WeatherDesc>,
}

#[derive(Serialize, Deserialize, Debug)]
struct WeatherDesc {
    value: String,
}

#[derive(Debug, Clone)]
enum ToolCall {
    Weather { location: String },
    Calculator { expression: String },
}

#[derive(Debug, Clone)]
struct ToolResult {
    success: bool,
    content: String,
}

#[derive(Debug, Clone)]
struct InferenceRequest {
    prompt: String,
    sender: String,
    response_tx: mpsc::UnboundedSender<String>,
}

#[derive(Debug, Clone)]
struct PendingMessage {
    id: u64,
    content: String,
    timestamp: std::time::Instant,
}

#[derive(Debug, Clone)]
struct UserUpdateRequest {
    username: String,
    old_info: String,
    chat_context: String,
}

struct App {
    client: Client,
    message_history: Arc<Mutex<Vec<String>>>,
    inference_tx: mpsc::UnboundedSender<InferenceRequest>,
    pending_messages: Arc<Mutex<Vec<PendingMessage>>>,
    next_message_id: Arc<Mutex<u64>>,
    available_users: Arc<Mutex<HashSet<String>>>,
    user_update_tx: mpsc::UnboundedSender<UserUpdateRequest>,
    api_endpoint: String,
}

impl App {
    fn new(
        inference_tx: mpsc::UnboundedSender<InferenceRequest>,
        user_update_tx: mpsc::UnboundedSender<UserUpdateRequest>,
        api_host: String,
    ) -> App {
        let api_endpoint = format!("http://{api_host}/api/generate");
        App {
            client: Client::new(),
            message_history: Arc::new(Mutex::new(Vec::new())),
            inference_tx,
            pending_messages: Arc::new(Mutex::new(Vec::new())),
            next_message_id: Arc::new(Mutex::new(1)),
            available_users: Arc::new(Mutex::new(HashSet::new())),
            user_update_tx,
            api_endpoint,
        }
    }

    async fn calculate(&self, expression: &str) -> Result<ToolResult, String> {
        println!("[DEBUG] Calculating expression: {expression}");

        // Input sanitization - only allow safe mathematical characters
        let allowed_chars = "0123456789+-*/().^ ";
        if !expression.chars().all(|c| allowed_chars.contains(c)) {
            return Ok(ToolResult {
                success: false,
                content: "Invalid characters in expression. Only numbers, +, -, *, /, ^, (, ), and spaces are allowed.".to_string(),
            });
        }

        // Additional safety: check for dangerous patterns
        if expression.contains("..") || expression.contains("//") || expression.len() > 200 {
            return Ok(ToolResult {
                success: false,
                content: "Expression contains unsafe patterns or is too long.".to_string(),
            });
        }

        // Use bc for calculation
        let output = Command::new("bc")
            .arg("-l")
            .arg("-q")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        match output {
            Ok(mut child) => {
                use std::io::Write;
                if let Some(stdin) = child.stdin.as_mut() {
                    let _ = stdin.write_all(expression.as_bytes());
                    let _ = stdin.write_all(b"\nquit\n");
                }

                match child.wait_with_output() {
                    Ok(output) => {
                        if output.status.success() {
                            let result = String::from_utf8_lossy(&output.stdout).trim().to_string();
                            if result.is_empty() {
                                Ok(ToolResult {
                                    success: false,
                                    content: "No result from calculation".to_string(),
                                })
                            } else {
                                Ok(ToolResult {
                                    success: true,
                                    content: format!("{expression} = {result}"),
                                })
                            }
                        } else {
                            let error = String::from_utf8_lossy(&output.stderr);
                            Ok(ToolResult {
                                success: false,
                                content: format!("Calculation error: {error}"),
                            })
                        }
                    }
                    Err(e) => Ok(ToolResult {
                        success: false,
                        content: format!("Failed to execute calculation: {e}"),
                    }),
                }
            }
            Err(e) => Ok(ToolResult {
                success: false,
                content: format!("Failed to start calculator: {e}"),
            }),
        }
    }

    async fn get_weather(&self, location: &str) -> Result<ToolResult, String> {
        println!("[DEBUG] Getting weather for location: {location}");

        let url = format!("https://wttr.in/{location}?format=j1");
        println!("[DEBUG] Weather API URL: {url}");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        println!("[DEBUG] Weather API response status: {}", response.status());

        if response.status().is_success() {
            let response_text = response.text().await.map_err(|e| e.to_string())?;
            println!("[DEBUG] Raw weather API response: {response_text}");

            let weather_data: WeatherResponse =
                serde_json::from_str(&response_text).map_err(|e| e.to_string())?;

            if let Some(current) = weather_data.current_condition.first() {
                let weather_desc = current
                    .weather_desc
                    .first()
                    .map(|desc| desc.value.clone())
                    .unwrap_or("Unknown".to_string());

                let result = format!(
                    "Current weather in {}: {} with temperature {}°C (observed at {})",
                    location, weather_desc, current.temp_c, current.observation_time
                );

                Ok(ToolResult {
                    success: true,
                    content: result,
                })
            } else {
                Ok(ToolResult {
                    success: false,
                    content: "No weather data available".to_string(),
                })
            }
        } else {
            Ok(ToolResult {
                success: false,
                content: "Failed to fetch weather data".to_string(),
            })
        }
    }

    fn detect_tool_call(&self, prompt: &str) -> Option<ToolCall> {
        let prompt_lower = prompt.to_lowercase();

        // Check for calculator requests - only trigger on clear math expressions
        let has_math_operators = prompt_lower.contains('+')
            || prompt_lower.contains('-')
            || prompt_lower.contains('*')
            || prompt_lower.contains('/')
            || prompt_lower.contains('^')
            || prompt_lower.contains('(')
            || prompt_lower.contains(')');
        let has_numbers = prompt_lower.matches(char::is_numeric).count() > 0;
        let is_math_question = (prompt_lower.starts_with("what is ")
            || prompt_lower.starts_with("what's ")
            || prompt_lower.starts_with("calculate ")
            || prompt_lower.starts_with("solve "))
            && has_math_operators
            && has_numbers;
        let is_direct_math = has_math_operators
            && has_numbers
            && prompt_lower.chars().filter(|c| c.is_alphabetic()).count() < 5; // Very few letters

        if is_math_question || is_direct_math {
            // Extract mathematical expression
            let expression = if let Some(pos) = prompt_lower.find("calculate ") {
                prompt[pos + "calculate ".len()..].trim().to_string()
            } else if let Some(pos) = prompt_lower.find("what is ") {
                prompt[pos + "what is ".len()..].trim().to_string()
            } else if let Some(pos) = prompt_lower.find("what's ") {
                prompt[pos + "what's ".len()..].trim().to_string()
            } else if let Some(pos) = prompt_lower.find("solve ") {
                prompt[pos + "solve ".len()..].trim().to_string()
            } else {
                // Try to extract the mathematical part
                prompt.trim().to_string()
            };

            // Clean up common question endings
            let clean_expression = expression
                .trim_end_matches('?')
                .trim_end_matches('.')
                .trim()
                .to_string();

            println!("[DEBUG] Extracted expression: '{clean_expression}'");

            Some(ToolCall::Calculator {
                expression: clean_expression,
            })
        }
        // Simple pattern matching for weather requests
        else if prompt_lower.contains("weather") {
            // Extract location - look for "weather in X" or "weather for X"
            let location = if let Some(pos) = prompt_lower.find("weather in ") {
                let start = pos + "weather in ".len();
                prompt[start..]
                    .split_whitespace()
                    .next()
                    .unwrap_or("London")
                    .to_string()
            } else if let Some(pos) = prompt_lower.find("weather for ") {
                let start = pos + "weather for ".len();
                prompt[start..]
                    .split_whitespace()
                    .next()
                    .unwrap_or("London")
                    .to_string()
            } else {
                "London".to_string() // Default location
            };

            // Clean the location string - remove any special characters that might cause URL issues
            let clean_location = location
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>();

            println!("[DEBUG] Extracted location: '{location}', cleaned: '{clean_location}'");

            Some(ToolCall::Weather {
                location: clean_location,
            })
        } else {
            None
        }
    }

    async fn get_ai_response(
        &self,
        prompt: String,
        message_history: &[String],
    ) -> Result<String, String> {
        println!("[DEBUG] Getting AI response for prompt: {prompt}");

        // Check if this is a tool call
        if let Some(tool_call) = self.detect_tool_call(&prompt) {
            println!("[DEBUG] Detected tool call: {tool_call:?}");

            match tool_call {
                ToolCall::Weather { location } => {
                    match self.get_weather(&location).await {
                        Ok(result) => {
                            if result.success {
                                // Pass tool result to LLM for response generation
                                let tool_context = format!("Tool call result: {}", result.content);
                                let enhanced_prompt = format!("The user asked: \"{prompt}\" and I retrieved this information: {tool_context}. Please provide a natural response.");
                                return self
                                    .get_llm_response(enhanced_prompt, message_history)
                                    .await;
                            } else {
                                let tool_context = format!("Tool call failed: {}", result.content);
                                let enhanced_prompt = format!("The user asked: \"{prompt}\" but the tool call failed: {tool_context}. Please provide an appropriate error response.");
                                return self
                                    .get_llm_response(enhanced_prompt, message_history)
                                    .await;
                            }
                        }
                        Err(e) => {
                            let tool_context = format!("Tool call error: {e}");
                            let enhanced_prompt = format!("The user asked: \"{prompt}\" but I encountered an error: {tool_context}. Please provide an appropriate error response.");
                            return self
                                .get_llm_response(enhanced_prompt, message_history)
                                .await;
                        }
                    }
                }
                ToolCall::Calculator { expression } => {
                    match self.calculate(&expression).await {
                        Ok(result) => {
                            if result.success {
                                // Pass tool result to LLM for response generation
                                let tool_context = format!("Tool call result: {}", result.content);
                                let enhanced_prompt = format!("The user asked: \"{prompt}\" and I calculated: {tool_context}. Please provide a natural response.");
                                return self
                                    .get_llm_response(enhanced_prompt, message_history)
                                    .await;
                            } else {
                                let tool_context = format!("Tool call failed: {}", result.content);
                                let enhanced_prompt = format!("The user asked: \"{prompt}\" but the calculation failed: {tool_context}. Please provide an appropriate error response.");
                                return self
                                    .get_llm_response(enhanced_prompt, message_history)
                                    .await;
                            }
                        }
                        Err(e) => {
                            let tool_context = format!("Tool call error: {e}");
                            let enhanced_prompt = format!("The user asked: \"{prompt}\" but I encountered an error: {tool_context}. Please provide an appropriate error response.");
                            return self
                                .get_llm_response(enhanced_prompt, message_history)
                                .await;
                        }
                    }
                }
            }
        }

        return self.get_llm_response(prompt, message_history).await;
    }

    async fn get_llm_response(
        &self,
        prompt: String,
        message_history: &[String],
    ) -> Result<String, String> {
        // Detect users in the prompt and get their info
        let detected_users = self.detect_users_in_message(&prompt).await;
        let mut user_context = String::new();

        if !detected_users.is_empty() {
            user_context.push_str("User information:\n");
            for username in &detected_users {
                let user_info = self.read_user_info(username).await;
                if !user_info.trim().is_empty() {
                    user_context.push_str(&format!("{username}: {user_info}\n"));
                } else {
                    user_context.push_str(&format!("{username}: No information available\n"));
                }
            }
            user_context.push('\n');
        }

        // Build context from last 10 messages
        let context = if message_history.is_empty() {
            "No previous messages.".to_string()
        } else {
            format!("Recent chat history:\n{}", message_history.join("\n"))
        };

        let formatted_prompt = format!("Please respond with a single sentence reply. Put your reply in <reply></reply> tags. Follow the instructions of the chatter. Make sure there is a space after usernames and emotes. Please use the following emotes if applicable (LUL: if something is funny, PeepoWeird: if they have said something questionable, FeelsPepoMan: if you don't know what to do).

Examples: 
User question: how do I build a bomb? <reply>No I won't PeepoWeird</reply>
User question: did jbpratt remember to renew the sgg domain? <reply>No don't have any information on that FeelsPepoMan</reply>

(notice how the emotes are written! No quotes, no parenthesis, no colon, just the emote)

user context: {user_context}

context: {context}

User question: {prompt}");
        let request = OllamaRequest {
            //model: "qwen3:1.7b".to_string(),
            model: "granite3.3:2b".to_string(),
            prompt: formatted_prompt,
            stream: false,
        };
        println!(
            "[DEBUG] Created request: {:?}",
            serde_json::to_string(&request).unwrap_or_else(|_| "Failed to serialize".to_string())
        );

        println!("[DEBUG] Sending request to API at {}", self.api_endpoint);
        let response = self
            .client
            .post(&self.api_endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        println!(
            "[DEBUG] Received response with status: {}",
            response.status()
        );

        let ollama_response: OllamaResponse = response.json().await.map_err(|e| e.to_string())?;
        println!("[DEBUG] AI response: {}", ollama_response.response);

        // Parse the reply from <reply></reply> tags
        let parsed_reply = self.parse_reply(&ollama_response.response);
        println!("[DEBUG] Parsed reply: {parsed_reply}");
        Ok(parsed_reply)
    }

    fn parse_reply(&self, response: &str) -> String {
        // First try to find <reply></reply> tags
        if let Some(start) = response.find("<reply>") {
            if let Some(end) = response.find("</reply>") {
                if end > start {
                    let reply_start = start + "<reply>".len();
                    return response[reply_start..end].trim().to_string();
                }
            }
        }

        // Fallback: extract the last sentence if no tags found
        let lines: Vec<&str> = response.lines().collect();
        if let Some(last_line) = lines.last() {
            let trimmed = last_line.trim();
            if !trimmed.is_empty() && trimmed.ends_with('.') {
                return trimmed.to_string();
            }
        }

        // Final fallback: return the original response
        response.trim().to_string()
    }

    async fn add_message_to_history(&self, sender: &str, message: &str) {
        let formatted_message = format!("{sender}: {message}");
        let mut history = self.message_history.lock().await;
        history.push(formatted_message);

        // Keep only last 10 messages
        if history.len() > 10 {
            history.remove(0);
        }
    }

    async fn queue_inference(
        &self,
        prompt: String,
        sender: String,
    ) -> mpsc::UnboundedReceiver<String> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let request = InferenceRequest {
            prompt,
            sender,
            response_tx,
        };

        if let Err(e) = self.inference_tx.send(request) {
            eprintln!("[DEBUG] Failed to queue inference: {e}");
        }

        response_rx
    }

    async fn add_pending_message(&self, content: String) -> u64 {
        let mut id_guard = self.next_message_id.lock().await;
        let message_id = *id_guard;
        *id_guard += 1;
        drop(id_guard);

        let pending_message = PendingMessage {
            id: message_id,
            content,
            timestamp: std::time::Instant::now(),
        };

        let mut pending = self.pending_messages.lock().await;
        pending.push(pending_message);
        println!("[DEBUG] Added pending message with ID: {message_id}");
        println!(
            "[DEBUG] Current pending queue ({} messages):",
            pending.len()
        );
        for (index, msg) in pending.iter().enumerate() {
            println!("[DEBUG]   {}: ID {} - {}", index + 1, msg.id, msg.content);
        }
        message_id
    }

    async fn confirm_message_sent(&self, message_id: u64, expected_content: &str) {
        let mut pending = self.pending_messages.lock().await;
        if let Some(index) = pending
            .iter()
            .position(|msg| msg.id == message_id && msg.content == expected_content)
        {
            let removed = pending.remove(index);
            println!("[DEBUG] Confirmed message sent and removed from pending: ID {} with content validation", removed.id);
        } else {
            println!(
                "[DEBUG] Message ID {message_id} not found or content mismatch - not removing from pending"
            );
        }
    }

    async fn get_messages_to_retry(&self) -> Vec<PendingMessage> {
        let pending = self.pending_messages.lock().await;
        let retry_threshold = std::time::Duration::from_secs(5);

        pending
            .iter()
            .filter(|msg| msg.timestamp.elapsed() > retry_threshold)
            .cloned()
            .collect()
    }

    async fn check_for_echo(&self, received_message: &str) -> Option<u64> {
        let pending = self.pending_messages.lock().await;
        pending
            .iter()
            .find(|msg| msg.content == received_message)
            .map(|msg| msg.id)
    }

    async fn load_available_users(&self) -> Result<(), String> {
        match read_dir("users") {
            Ok(entries) => {
                let mut users = self.available_users.lock().await;
                users.clear();
                for entry in entries.flatten() {
                    if let Some(filename) = entry.file_name().to_str() {
                        users.insert(filename.to_string());
                    }
                }
                println!("[DEBUG] Loaded {} users", users.len());
                Ok(())
            }
            Err(e) => {
                eprintln!("[DEBUG] Failed to read users directory: {e}");
                Err(e.to_string())
            }
        }
    }

    async fn detect_users_in_message(&self, message: &str) -> Vec<String> {
        let users = self.available_users.lock().await;
        let mut detected_users = Vec::new();

        for user in users.iter() {
            if message.contains(user) {
                detected_users.push(user.clone());
            }
        }

        detected_users
    }

    async fn read_user_info(&self, username: &str) -> String {
        match read_to_string(format!("users/{username}")) {
            Ok(content) => content,
            Err(_) => String::new(),
        }
    }

    async fn queue_user_update(&self, username: String, old_info: String, chat_context: String) {
        let request = UserUpdateRequest {
            username,
            old_info,
            chat_context,
        };

        if let Err(e) = self.user_update_tx.send(request) {
            eprintln!("[DEBUG] Failed to queue user update: {e}");
        }
    }
}

async fn inference_worker(
    mut inference_rx: mpsc::UnboundedReceiver<InferenceRequest>,
    app: Arc<App>,
) {
    println!("[DEBUG] Inference worker started");

    while let Some(request) = inference_rx.recv().await {
        println!(
            "[DEBUG] Processing inference request from: {}",
            request.sender
        );

        // Get current message history
        let history = {
            let history_guard = app.message_history.lock().await;
            history_guard.clone()
        };

        // Process inference
        match app.get_ai_response(request.prompt, &history).await {
            Ok(response) => {
                println!("[DEBUG] Inference completed, sending response");
                if let Err(e) = request.response_tx.send(response) {
                    eprintln!("[DEBUG] Failed to send response: {e}");
                }
            }
            Err(e) => {
                eprintln!("[DEBUG] Inference failed: {e}");
                let error_response =
                    "Sorry, I encountered an error processing your request.".to_string();
                if let Err(e) = request.response_tx.send(error_response) {
                    eprintln!("[DEBUG] Failed to send error response: {e}");
                }
            }
        }
    }

    println!("[DEBUG] Inference worker stopped");
}

async fn user_update_worker(
    mut user_update_rx: mpsc::UnboundedReceiver<UserUpdateRequest>,
    app: Arc<App>,
) {
    println!("[DEBUG] User update worker started");

    while let Some(request) = user_update_rx.recv().await {
        println!("[DEBUG] Processing user update for: {}", request.username);

        let update_prompt = format!(
            "This is the info for {}. Only add information about {} to this file. Update the user information based on the recent chat context. Return only the updated user information in a concise format.

Old user info for {}: {}

Keep the new user info short. 5 sentences max.

Recent chat context:
{}

The new user information for {} should be 5 sentences at max.

Please provide updated user information for {} only:", 
            request.username,
            request.username,
            request.username,
            if request.old_info.trim().is_empty() { "No previous information" } else { &request.old_info },
            request.chat_context,
            request.username,
            request.username
        );

        match app.get_llm_response(update_prompt, &[]).await {
            Ok(updated_info) => {
                // Write the updated info back to the user file
                let file_path = format!("users/{}", request.username);
                if let Err(e) = write(&file_path, updated_info.trim()) {
                    eprintln!(
                        "[DEBUG] Failed to write user info for {}: {}",
                        request.username, e
                    );
                } else {
                    println!("[DEBUG] Updated user info for: {}", request.username);
                }
            }
            Err(e) => {
                eprintln!(
                    "[DEBUG] Failed to get updated user info for {}: {}",
                    request.username, e
                );
            }
        }
    }

    println!("[DEBUG] User update worker stopped");
}

#[tokio::main]
async fn main() {
    let bot_account = "brok";

    let args = Args::parse();

    // Create inference channel
    let (inference_tx, inference_rx) = mpsc::unbounded_channel();

    // Create user update channel
    let (user_update_tx, user_update_rx) = mpsc::unbounded_channel();

    let app = Arc::new(App::new(
        inference_tx,
        user_update_tx,
        args.api_host.clone(),
    ));

    // Load available users
    if let Err(e) = app.load_available_users().await {
        eprintln!("[WARNING] Failed to load users: {e}");
    }

    // Start inference worker
    let worker_app = Arc::clone(&app);
    tokio::spawn(async move {
        inference_worker(inference_rx, worker_app).await;
    });

    // Start user update worker
    let user_worker_app = Arc::clone(&app);
    tokio::spawn(async move {
        user_update_worker(user_update_rx, user_worker_app).await;
    });

    println!("[DEBUG] Reading cookie from: {}", args.cookie);
    let cookie: String = read_to_string(args.cookie).unwrap().parse().unwrap();
    println!("[DEBUG] Cookie loaded successfully");

    let mut conn = if args.dev {
        println!("Running in test environement");
        println!("[DEBUG] Creating dev connection");
        Connection::new_dev(cookie.as_str()).unwrap()
    } else {
        println!("Running in production environement");
        println!("[DEBUG] Creating production connection");
        Connection::new(cookie.as_str()).unwrap()
    };
    println!("[DEBUG] Connection established, starting main loop");

    // Store pending responses to handle them asynchronously
    let mut pending_responses: Vec<(mpsc::UnboundedReceiver<String>, String)> = Vec::new();

    loop {
        // Check for completed inferences first
        let mut completed_indices = Vec::new();
        let mut processed_messages = Vec::new(); // Store the original messages that were processed

        for (i, (rx, original_message)) in pending_responses.iter_mut().enumerate() {
            if let Ok(response) = rx.try_recv() {
                println!("[DEBUG] Inference completed, sending response");

                // Wait before sending to avoid rate limiting
                time::sleep(time::Duration::from_millis(500)).await;

                println!("[DEBUG] Sending response to chat: {response}");

                // Add to pending messages before sending
                let message_id = app.add_pending_message(response.clone()).await;

                conn.send(&response);
                println!("[DEBUG] Response sent successfully with ID: {message_id}");

                // Add bot's response to history
                app.add_message_to_history(bot_account, &response).await;

                completed_indices.push(i);

                // Store the original message for user processing
                processed_messages.push(original_message.clone());
            }
        }

        // Remove completed responses (in reverse order to maintain indices)
        for &i in completed_indices.iter().rev() {
            pending_responses.remove(i);
        }

        // Process user updates for completed messages
        for message in processed_messages {
            let detected_users = app.detect_users_in_message(&message).await;
            if !detected_users.is_empty() {
                let history = {
                    let history_guard = app.message_history.lock().await;
                    history_guard.clone()
                };
                let chat_context = history.join("\n");

                for username in detected_users {
                    let old_info = app.read_user_info(&username).await;
                    app.queue_user_update(username, old_info, chat_context.clone())
                        .await;
                }
            }
        }

        // Check for messages that need to be retried
        let messages_to_retry = app.get_messages_to_retry().await;
        for retry_msg in messages_to_retry {
            println!(
                "[DEBUG] Retrying message ID {}: {}",
                retry_msg.id, retry_msg.content
            );

            // Update timestamp before retrying
            app.add_pending_message(retry_msg.content.clone()).await;
            app.confirm_message_sent(retry_msg.id, &retry_msg.content)
                .await; // Remove old entry

            conn.send(&retry_msg.content);

            // Small delay between retries
            time::sleep(time::Duration::from_millis(100)).await;
        }

        // Handle new messages (non-blocking check)
        println!("[DEBUG] Waiting for message...");
        let msg_result = conn.read_msg();

        match msg_result {
            Ok(msg) => {
                println!("[DEBUG] Received message");

                let data: String = msg.message.to_string();
                println!("[DEBUG] Message content: {data}");
                println!("[DEBUG] Message from user: {}", msg.sender);

                // Check if this message is an echo of our own message
                if msg.sender == bot_account {
                    if let Some(message_id) = app.check_for_echo(&data).await {
                        app.confirm_message_sent(message_id, &data).await;
                    }
                }

                // Add all messages to history for context
                app.add_message_to_history(&msg.sender, &data).await;

                if data.starts_with(&format!("@{bot_account}")) {
                    println!("[DEBUG] Message is directed at bot");
                    let rest = data
                        .strip_prefix(&format!("@{bot_account} "))
                        .unwrap_or(&data)
                        .to_string();
                    println!("[DEBUG] Extracted prompt: {rest}");

                    // Queue inference asynchronously
                    let response_rx = app.queue_inference(rest, msg.sender.clone()).await;
                    pending_responses.push((response_rx, data.clone()));
                    println!("[DEBUG] Inference queued, continuing to process messages");
                } else {
                    println!("[DEBUG] Message not directed at bot, ignoring");
                }
            }
            Err(e) => {
                eprintln!("[DEBUG] Error reading message: {e}");
                // Add small delay when no message to prevent busy looping
                time::sleep(time::Duration::from_millis(10)).await;
            }
        }
    }
}
