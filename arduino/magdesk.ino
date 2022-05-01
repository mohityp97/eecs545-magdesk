#include "Adafruit_MLX90393_new.h"
#include <Adafruit_TinyUSB.h>

#define ARRAY_ID 0x00

//#define DEBUG
//#define DEBUG_TIME
#define HANDSHAKE
#define SEND_SAMPLES
#define NUM_SENSORS         16
#define START_BYTE_1_SAMPLE 0x42
#define START_BYTE_2_SAMPLE 0x4D
#define I2C_CLOCK_SPEED     400000
#define UART_BAUD_RATE      921600
//#define UART_BAUD_RATE             115200

typedef struct MAGNETOMETER_SENSOR_STRUCT
{
    int16_t x;
    int16_t y;
    int16_t z;
} MAGNETOMETER_SENSOR;

typedef struct MAGNETOMETER_ARRAY_STRUCT  
{
    MAGNETOMETER_SENSOR sample[NUM_SENSORS];
    unsigned long int relative_timestamp_us = 0;
} MAGNETOMETER_ARRAY;

const uint16_t sensor_i2c_addresses[NUM_SENSORS] = {
    0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B};

Adafruit_MLX90393 sensor[NUM_SENSORS];
unsigned long int first_start_time_us = 0;
MAGNETOMETER_ARRAY magnetometer_array;

bool start_sample_collection()
{
    for (uint16_t sensor_iter = 0; sensor_iter < NUM_SENSORS; sensor_iter++)
    {
        if (sensor[sensor_iter].startSingleMeasurement() == false)
        {
            return false;
        }
    }
    return true;
}

bool collect_samples_restart_collection()
{
    sensors_event_t event;
    magnetometer_array.relative_timestamp_us = micros() - first_start_time_us;
    for (uint16_t sensor_iter = 0; sensor_iter < NUM_SENSORS; sensor_iter++)
    {
        if (!sensor[sensor_iter].getEvent(&event))
        {
            return false;
        }
#ifdef DEBUG
          Serial.println("Magnetometer Data:");
          Serial.print("Sensor Number:");
          Serial.print(sensor_iter);
          Serial.print("\tx:");
          Serial.print(event.magnetic.x);
          Serial.print("\ty: ");
          Serial.print(event.magnetic.y);
          Serial.print("\tz: ");
          Serial.println(event.magnetic.z); 
#endif
        magnetometer_array.sample[sensor_iter].x = (int16_t)event.magnetic.x;
        magnetometer_array.sample[sensor_iter].y = (int16_t)event.magnetic.y;
        magnetometer_array.sample[sensor_iter].z = (int16_t)event.magnetic.z;

        sensor[sensor_iter].startSingleMeasurement();
    }
    return true;
}

void send_samples_uart()
{
    uint16_t checksum = 0;
    // Start Byte 1 for a sample from N sensors
    Serial.write(START_BYTE_1_SAMPLE);
    // Start Byte 2 for a sample from N sensors
    Serial.write(START_BYTE_2_SAMPLE);

    // Array ID
    Serial.write((uint8_t)ARRAY_ID);
    // Number of sensors in the array
    Serial.write((uint8_t)NUM_SENSORS);

    // Time stamp
    Serial.write((uint8_t)((magnetometer_array.relative_timestamp_us & 0xFF000000) >> 24));
    Serial.write((uint8_t)((magnetometer_array.relative_timestamp_us & 0x00FF0000) >> 16));
    Serial.write((uint8_t)((magnetometer_array.relative_timestamp_us & 0x0000FF00) >> 8));
    Serial.write((uint8_t)(magnetometer_array.relative_timestamp_us & 0x000000FF));

    for (uint16_t sensor_iter = 0; sensor_iter < NUM_SENSORS; sensor_iter++)
    {
        checksum = 0;

        Serial.write(magnetometer_array.sample[sensor_iter].x >> 8);
        Serial.write((int8_t)magnetometer_array.sample[sensor_iter].x);

        checksum += abs(magnetometer_array.sample[sensor_iter].x);

        Serial.write(magnetometer_array.sample[sensor_iter].y >> 8);
        Serial.write((uint8_t)magnetometer_array.sample[sensor_iter].y);

        checksum += abs(magnetometer_array.sample[sensor_iter].y);

        Serial.write(magnetometer_array.sample[sensor_iter].z >> 8);
        Serial.write((uint8_t)magnetometer_array.sample[sensor_iter].z);

        checksum += abs(magnetometer_array.sample[sensor_iter].z);

        Serial.write(checksum >> 8);
        Serial.write((uint8_t)checksum);
    }
}

void setup(void)
{
    first_start_time_us = micros();
    Serial.begin(UART_BAUD_RATE);
//    Serial.println(" ");
    Wire.setClock(I2C_CLOCK_SPEED);

    while (!Serial)
    {
        delay(10);
    }

#ifdef DEBUG
    Serial.println("Starting Adafruit MLX90393 Demo");
#endif

    for (uint16_t sensor_iter = 0; sensor_iter < NUM_SENSORS; sensor_iter++)
    {
        sensor[sensor_iter] = Adafruit_MLX90393();
        if (!sensor[sensor_iter].begin_I2C(sensor_i2c_addresses[sensor_iter]))
        {
#ifdef DEBUG
            Serial.println("No sensor found ... check your wiring?");
#endif
            while (1)
            {
                delay(10);
            }
        }
#ifdef DEBUG
        Serial.println("Found a MLX90393 sensor");
#endif
        sensor[sensor_iter].set_I2C_Speed(I2C_CLOCK_SPEED);

        sensor[sensor_iter].setGain(MLX90393_GAIN_5X);

#ifdef DEBUG
        Serial.print("Gain set to: ");
        switch (sensor[sensor_iter].getGain())
        {
            case MLX90393_GAIN_1X:
                Serial.println("1 x");
                break;
            case MLX90393_GAIN_1_33X:
                Serial.println("1.33 x");
                break;
            case MLX90393_GAIN_1_67X:
                Serial.println("1.67 x");
                break;
            case MLX90393_GAIN_2X:
                Serial.println("2 x");
                break;
            case MLX90393_GAIN_2_5X:
                Serial.println("2.5 x");
                break;
            case MLX90393_GAIN_3X:
                Serial.println("3 x");
                break;
            case MLX90393_GAIN_4X:
                Serial.println("4 x");
                break;
            case MLX90393_GAIN_5X:
                Serial.println("5 x");
                break;
        }
#endif

//        sensor[sensor_iter].setHallConf(MLX90393_HALL_CONF_0C);
#ifdef DEBUG
        Serial.print("Hall Conf set to: ");
        switch (sensor[sensor_iter].getHallConf())
        {
            case MLX90393_HALL_CONF_00:
                Serial.println("0x00");
                break;
            case MLX90393_HALL_CONF_0C:
                Serial.println("0x0C");
                break;
        }
#endif

//        sensor[sensor_iter].setResolution(MLX90393_X, MLX90393_RES_19);
//        sensor[sensor_iter].setResolution(MLX90393_Y, MLX90393_RES_19);
//        sensor[sensor_iter].setResolution(MLX90393_Z, MLX90393_RES_16);

        sensor[sensor_iter].setOversampling(MLX90393_OSR_0);

        sensor[sensor_iter].setFilter(MLX90393_FILTER_2);
    }
#ifdef HANDSHAKE
    while(!Serial.available())
    {
      delay(10);
    }
    while(Serial.available())
    {
      Serial.read();
    }
    while(!Serial.available())
    {
      Serial.write(0x52);
      Serial.write(0x5D);
    }
#endif
    

    while (!start_sample_collection())
    {
        delay(10);
    }
}

void loop(void)
{
#ifdef DEBUG_TIME
    long long int start_time   = micros();
    long long int elapsed_time = micros();
#endif
    if (collect_samples_restart_collection())
    {
#ifdef SEND_SAMPLES
        send_samples_uart();
#endif

#ifdef DEBUG_TIME
        elapsed_time = micros() - start_time;
        Serial.print("Sampling Frequency:\t");
        Serial.print((1000000.0f / (float)elapsed_time));
        Serial.println(" Hz");
#endif
    }
}
