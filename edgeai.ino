#include <SPI.h>
#include <LibLacuna.h>
#include <I2S.h>
#include "birds_model_fixed.h"
#include "secrets.h"

#ifndef REGION
#define REGION R_EU868
#endif

static lsLoraWANParams loraWANParams;
static lsLoraTxParams txParams;

#define I2S_SAMPLE_RATE 16000  // [16000, 48000] supported by the microphone
#define I2S_BITS_PER_SAMPLE 32 // Data is sent in 32-bit packets over I2S but only 18 bits are used by the microphone, remaining least significant bits are set to 0

static input_t inputs; // 1-channel, 16000 samples for 16kHz over 1s
static volatile size_t sample_i = 0;
static output_t outputs;
static volatile boolean ready_for_inference = false;

const char* BIRD_CLASSES[] = {"Alauda arvensis", "Emberiza cirlus", "Muscicapa striata"};

void processI2SData(uint8_t *data, size_t size) {
    int32_t *data32 = (int32_t*)data;

    // Copy first channel into model inputs
    size_t i = 0;
    for (i = 0; i < size / 8 && sample_i + i < MODEL_INPUT_DIM_0; i++, sample_i++) {
      inputs[sample_i][0] = data32[i * 2] >> 14; // Drop 32 - 18 = 14 unused bits
    }

    if (sample_i >= MODEL_INPUT_DIM_0) {
      ready_for_inference = true;
    }
}

void onI2SReceive() {
  size_t size = I2S.available();
  static uint8_t data[I2S_BUFFER_SIZE];

  if (size > 0) {
    I2S.read(data, size);
    processI2SData(data, size);
  }
}

void setup() {
  Serial.begin(115200);

  // For RFThing-DKAIoT
  pinMode(PIN_LED, OUTPUT);
  pinMode(LS_GPS_ENABLE, OUTPUT);
  digitalWrite(LS_GPS_ENABLE, LOW);
  pinMode(LS_GPS_V_BCKP, OUTPUT);
  digitalWrite(LS_GPS_V_BCKP, LOW);
  pinMode(SD_ON_OFF, OUTPUT);
  digitalWrite(SD_ON_OFF, HIGH);

  delay(100); // Wait for peripheral power rail to stabilize after setting SD_ON_OFF

  // start I2S
  if (!I2S.begin(I2S_PHILIPS_MODE, I2S_SAMPLE_RATE, I2S_BITS_PER_SAMPLE, false)) {
    Serial.println("Failed to initialize I2S!");
    while (1); // do nothing
  }

  I2S.onReceive(onI2SReceive);

  // Trigger a read to start DMA
  I2S.peek();

  // LoRa
  // SX1262 configuration for lacuna LS200 board
  lsSX126xConfig cfg;
  lsCreateDefaultSX126xConfig(&cfg, BOARD_VERSION);

  // Special configuration for DKAIoT Board
  cfg.nssPin = E22_NSS;                           //19
  cfg.resetPin = E22_NRST;                        //14
  cfg.antennaSwitchPin = E22_RXEN;                //1
  cfg.busyPin = E22_BUSY;                         //2  
  cfg.dio1Pin = E22_DIO1;                         //39

  // Initialize SX1262  
  int result = lsInitSX126x(&cfg, REGION);
  
  // LoRaWAN session parameters
  lsCreateDefaultLoraWANParams(&loraWANParams, networkKey, appKey, deviceAddress);
  loraWANParams.txPort = 1;
  loraWANParams.rxEnable = true;

  // transmission parameters for terrestrial LoRa
  lsCreateDefaultLoraTxParams(&txParams, REGION);  
  txParams.spreadingFactor = lsLoraSpreadingFactor_7;
  txParams.frequency = 868100000;
}

void loop() {
  if (ready_for_inference) {
    // Input buffer full, perform inference
    
    // Turn LED on during preprocessing/prediction
    digitalWrite(PIN_LED, HIGH);

    // Start timer
    long long t_start = millis();
    
    // Compute DC offset 
    int32_t dc_offset = 0;

    for (size_t i = 0; i < sample_i; i++) { // Accumulate samples
      dc_offset += inputs[i][0];  
    }

    dc_offset = dc_offset / (int32_t)sample_i; // Compute average over samples
    
    // Filtering
    for (size_t i = 0; i < sample_i; i++) {
      // Remove DC offset
      inputs[i][0] -= dc_offset;

      // Amplify 
      inputs[i][0] = inputs[i][0] << 2;
    }
        
    // Predict
    cnn(inputs, outputs);

    // Get output class
    unsigned int label = 0;  
    int16_t max_val = outputs[0];
    for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
      if (max_val < outputs[i]) {
        max_val = outputs[i];
        label = i;
      }
    }
    
    // Check if a bird species was detected with high confidence
    if (label < 3 && max_val >= 32767.0f) {
      
      // Prepare data to send via LoRaWAN      
      static unsigned char data[4];
      data[0] = label;
      uint32_t confidence = (uint32_t)(max_val + 0.5f);       
      data[1] = (max_val >> 8) & 0xFF;  // high byte
      data[2] = max_val & 0xFF;         // low byte
            
      // Reconstruct confidence from bytes for display
      int16_t displayedConfidence = ((int16_t)data[1] << 8) | data[2];

      Serial.print("Bird species detected: ");
      Serial.print(BIRD_CLASSES[data[0]]);  
      Serial.print(", Confidence: ");
      Serial.print(displayedConfidence);
      
      // Send LoRaWAN message
      int lora_result = lsSendLoraWAN(&loraWANParams, &txParams, data, sizeof(data));
      
      Serial.print(", LORAWAN_RETURN_CODE: ");  
      Serial.println(lora_result);
    }
    
    // Turn LED off after prediction
    digitalWrite(PIN_LED, LOW);
    
    ready_for_inference = false;
    sample_i = 0;
  }
}