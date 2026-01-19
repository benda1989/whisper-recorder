#include <WiFi.h>
#include <driver/i2s.h>
#include <esp_sleep.h>
#include <WiFiManager.h>
#include <EEPROM.h>

#define I2S_WS 0//4
#define I2S_SCK 1//5
#define I2S_SD 2//8
typedef struct {
  char serverAddress[100];
  uint16_t serverPort;
  float audioGain;
} ConfigData;

ConfigData config;

String deviceId = String((uint32_t)ESP.getEfuseMac(), HEX);

// Load configuration
void loadConfig() {
    EEPROM.begin(sizeof(ConfigData) + 1);
    EEPROM.get(0, config);
    EEPROM.end();
    if (config.serverAddress[0] == 0 || config.serverPort == 0) {
        strcpy(config.serverAddress, "192.168.40.4");
        config.serverPort = 8883;
        config.audioGain = 6.0;
    }
}

// Save configuration
void saveConfig(WiFiManagerParameter* custom_server_address, WiFiManagerParameter* custom_server_port, WiFiManagerParameter* custom_audio_gain) {
    EEPROM.begin(sizeof(ConfigData) + 1);
    
    if (strlen(custom_server_address->getValue()) > 0) {
        strncpy(config.serverAddress, custom_server_address->getValue(), sizeof(config.serverAddress));
    }
    if (strlen(custom_server_port->getValue()) > 0) {
        config.serverPort = atoi(custom_server_port->getValue());
    }
    if (strlen(custom_audio_gain->getValue()) > 0) {
        config.audioGain = atof(custom_audio_gain->getValue());
    }
    
    EEPROM.put(0, config);
    EEPROM.commit();
    EEPROM.end();
    Serial.printf("Configuration saved - Server: %s:%d, Audio gain: %.1f\n", config.serverAddress, config.serverPort, config.audioGain);
}

// Try to connect to WiFi
bool tryConnectToWiFi() {
    Serial.print("Trying to connect to WiFi...");
    WiFi.mode(WIFI_STA);
    WiFi.begin();  // Use saved credentials
    
    int retries = 0;
    while (retries < 10 && WiFi.status() != WL_CONNECTED) {
        retries++;
        Serial.print(".");
        delay(1000);
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        return true;
    } else {
        return false;
    }
}

// Setup WiFiManager
void setupWiFiManager() {
    WiFiManager wm;
    WiFiManagerParameter custom_server_address("serverAddress", "Server Address", config.serverAddress, 100);
    WiFiManagerParameter custom_server_port("serverPort", "Server Port", String(config.serverPort).c_str(), 10);
    WiFiManagerParameter custom_audio_gain("audioGain", "Audio Gain", String(config.audioGain).c_str(), 10);
    wm.setConfigPortalTimeout(1800);
    wm.addParameter(&custom_server_address);
    wm.addParameter(&custom_server_port);
    wm.addParameter(&custom_audio_gain);

    // Start configuration portal
    Serial.println("Starting WiFi configuration portal...");
    Serial.print("AP name: ");

    String apName = "ESP32-" + deviceId;
    if (!wm.autoConnect(apName.c_str(), "1234567890")) {
        Serial.println("Configuration portal timeout");
    }else{
        saveConfig(&custom_server_address, &custom_server_port, &custom_audio_gain);
    }
}

void setup() {
    Serial.begin(115200);
    Serial.println("ESP32 Starting...");
    
    // Load configuration
    loadConfig();
    Serial.printf("Current configuration - Server: %s:%d, Audio gain: %.1f\n", config.serverAddress, config.serverPort, config.audioGain);
    
    // If WiFi connection fails, start WiFiManager
    if (!tryConnectToWiFi()) {
        Serial.println("WiFi connection failed, starting configuration portal...");
        setupWiFiManager();
        delay(1000);
        ESP.restart();
    }
    
    // Connect to server (TCP long connection)
    WiFiClient client;
    
    if (!client.connect(config.serverAddress, config.serverPort)) {
        Serial.printf("Failed to connect to server %s:%d, sleeping for 60 seconds\n", config.serverAddress, config.serverPort);
        esp_sleep_enable_timer_wakeup(60 * 1000000);
        esp_deep_sleep_start();
    }
    //WiFi.setTxPower(WIFI_POWER_17dBm); 
    // Send device ID to get recording duration
    client.print("device_id:" + deviceId);
    
    unsigned long timeout = millis() + 5000;
    while (!client.available() && millis() < timeout) {
        delay(100);
    }
    
    int recordingDuration = 0;
    if (client.available()) {
        String response = client.readStringUntil('\n');
        recordingDuration = response.toInt();
        Serial.printf("Server response: %d seconds\n", recordingDuration);
    }
    
    if (recordingDuration <= 0) {
        Serial.println("No recording schedule, sleeping for 60 seconds");
        client.stop();
        esp_sleep_enable_timer_wakeup(60 * 1000000);
        esp_deep_sleep_start();
    }
    
    // Initialize I2S
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 6,
        .dma_buf_len = 512
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    
    // Start recording using same connection (TCP long connection)
    uint8_t audioBuffer[2048];
    
    unsigned long targetEndTime = millis() + (recordingDuration * 1000);
    
    while (millis() < targetEndTime) {
        size_t bytesRead;
        esp_err_t readResult = i2s_read(I2S_NUM_0, audioBuffer, 2048, &bytesRead, 100);
        
        if (readResult == ESP_OK && bytesRead > 0) {
            // Apply audio gain
            int16_t* samples = (int16_t*)audioBuffer;
            int sampleCount = bytesRead / 2; // 16-bit samples
            
            for (int i = 0; i < sampleCount; i++) {
                int32_t amplified = (int32_t)samples[i] * config.audioGain;
                // Limit amplitude to prevent overflow
                if (amplified > 32767) amplified = 32767;
                else if (amplified < -32768) amplified = -32768;
                samples[i] = (int16_t)amplified;
            }
            
            if (client.connected()) {
                client.write(audioBuffer, bytesRead);
            } else {
                Serial.println("Client connection disconnected, stopping recording");
                break;
            }
        } 
        yield();
    }
    
    Serial.println("Recording completed, sleeping for 60 seconds");
    client.stop();
    i2s_driver_uninstall(I2S_NUM_0);
    WiFi.disconnect(true);
    esp_sleep_enable_timer_wakeup(60 * 1000000);
    esp_deep_sleep_start();
}

void loop() {
    // This will never be reached
}