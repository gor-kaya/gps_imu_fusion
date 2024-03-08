#include <Adafruit_GPS.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <Wire.h>
#include <utility/imumaths.h>
#include <SD.h>
#include <SoftwareSerial.h>
#include <SPI.h>

/* Set the delay between fresh IMU samples */
uint16_t BNO055_SAMPLERATE_DELAY_MS = 100;

// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

SoftwareSerial gpsSerial(3, 2); // RX, TX for GPS module
Adafruit_GPS GPS(&gpsSerial);
File gpsFile;
File imuFile;

const int chipSelect = 4; // CS pin for SD card

void setup() 
{
  Serial.begin(115200);
  while (!Serial) delay(10);  // wait for serial port to open!
  GPS.begin(9600);
  SD.begin(chipSelect);

  /* Initialise the IMU sensor */
  if (!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }

  GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
  // Set the update rate
  GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ);   // 1 Hz update rate

  gpsFile = SD.open("gps_data.txt", FILE_WRITE);
  if (gpsFile) {
      gpsFile.println("START OF NEW GPS RECORDING");
      gpsFile.close();
  } else {
      Serial.println("Error opening gps_data.txt");
  }

  imuFile = SD.open("gps_data.txt", FILE_WRITE);
  if (imuFile) {
      imuFile.println("START OF NEW IMU RECORDING");
      imuFile.close();
  } else {
      Serial.println("Error opening gps_data.txt");
  }
}

uint32_t timer = millis();
void loop()
{
  if (millis() - timer > 1000) {
    timer = millis(); //reset the timer

    gpsFile = SD.open("gps_data.txt", FILE_WRITE);
    if (gpsFile) {

      gpsFile.print("\nTime: ");
      if (GPS.hour < 10) { gpsFile.print('0'); }
      gpsFile.print(GPS.hour, DEC); gpsFile.print(':');
      if (GPS.minute < 10) { gpsFile.print('0'); }
      gpsFile.print(GPS.minute, DEC); gpsFile.print(':');
      if (GPS.seconds < 10) { gpsFile.print('0'); }
      gpsFile.print(GPS.seconds, DEC); gpsFile.print('.');
      if (GPS.milliseconds < 10) {
        gpsFile.print("00");
      } else if (GPS.milliseconds > 9 && GPS.milliseconds < 100) {
        gpsFile.print("0");
      }
      gpsFile.print("\nFix: "); gpsFile.print((int)GPS.fix);
      gpsFile.print(" quality: "); gpsFile.println((int)GPS.fixquality);
      if (GPS.fix) {
        gpsFile.print("Location: ");
        gpsFile.print(GPS.latitude, 4); gpsFile.print(GPS.lat);
        gpsFile.print(", ");
        gpsFile.print(GPS.longitude, 4); gpsFile.println(GPS.lon);
        gpsFile.print("Altitude: "); gpsFile.println(GPS.altitude);
        
        gpsFile.print("Speed (knots): "); gpsFile.println(GPS.speed);
        gpsFile.print("Angle: "); gpsFile.println(GPS.angle);
      }
      gpsFile.close();
    } else {
        Serial.println("Error opening gps file!");
    }

    imuFile = SD.open("gps_data.txt", FILE_WRITE);
    if (imuFile) {
      imuFile.println("IMU IS ON BICZES");
      
      imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
      imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
      imu::Vector<3> linearaccel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);

      imuFile.print("Euler: ");
      imuFile.print(euler.x()); imuFile.print(",");
      imuFile.print(euler.y()); imuFile.print(",");
      imuFile.print(euler.z()); imuFile.print(",");

      imuFile.close();
    } else {
        Serial.println("Error opening imu file!");
    }
  }
}