// This code records GPS PA1616D and IMU BNO055 data on to the SD card.
// For debugging set GPSECHO to true
// Saved file follows CSV format. One line of data is as follows:
// "Time (hh:mm:ss.ss), Latitude (DDMM.MMMM), Longitude (DDMM.MMMM), Altitude (MSL), SOG (knots), COG (degrees, 0=North), 
// euler x, y, z (degrees, based on 360* sphere), gyro x, y, z (angular velocity, rad/s), linear acceleration x, y, z (acceleration minus gravity, m/s^2)"

#include <Adafruit_GPS.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_LSM9DS1.h>
#include <Wire.h>
#include <SD.h>
#include <SoftwareSerial.h>
#include <SPI.h>

// BNO055
// /* Set the delay between fresh IMU samples */
// uint16_t BNO055_SAMPLERATE_DELAY_MS = 100;

// // Check I2C device address and correct line below (by default address is 0x29 or 0x28)
// //                                   id, address
// Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

// LSM9DS1
// i2c
Adafruit_LSM9DS1 lsm = Adafruit_LSM9DS1();

#define LSM9DS1_SCK A5
#define LSM9DS1_MISO 12
#define LSM9DS1_MOSI A4
#define LSM9DS1_XGCS 6
#define LSM9DS1_MCS 5

// GPS PA1616D
SoftwareSerial mySerial(3, 2); // RX, TX for GPS module
Adafruit_GPS GPS(&mySerial);
#define GPSECHO  false
File gpsFile;
File imuFile;

const int chipSelect = 10; // CS pin for SD card

void setupSensor() // for LSM9DS1
{
  // 1.) Set the accelerometer range
  lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_2G, lsm.LSM9DS1_ACCELDATARATE_10HZ);
  //lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_4G, lsm.LSM9DS1_ACCELDATARATE_119HZ);
  //lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_8G, lsm.LSM9DS1_ACCELDATARATE_476HZ);
  //lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_16G, lsm.LSM9DS1_ACCELDATARATE_952HZ);
  
  // 2.) Set the magnetometer sensitivity
  lsm.setupMag(lsm.LSM9DS1_MAGGAIN_4GAUSS);
  //lsm.setupMag(lsm.LSM9DS1_MAGGAIN_8GAUSS);
  //lsm.setupMag(lsm.LSM9DS1_MAGGAIN_12GAUSS);
  //lsm.setupMag(lsm.LSM9DS1_MAGGAIN_16GAUSS);

  // 3.) Setup the gyroscope
  lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_245DPS);
  //lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_500DPS);
  //lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_2000DPS);
}

void setup() 
{
  Serial.begin(115200);
  while (!Serial) delay(10);  // wait for serial port to open!
  SD.begin(chipSelect);

  /* Initialise the IMU sensor */
  if (!lsm.begin())
  {
    /* There was a problem detecting the LSM9DS1 ... check your connections */
    Serial.print("Oops ... unable to initialize the LSM9DS1. Check your wiring!");
    while (1);
  }
  setupSensor();
  delay(1000);
  GPS.begin(9600);

  GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
  // Set the update rate
  GPS.sendCommand(PMTK_SET_NMEA_UPDATE_5HZ);   // 5 Hz update rate
  // Request updates on antenna status, comment out to keep quiet
  GPS.sendCommand(PGCMD_ANTENNA);

  delay(1000);

  gpsFile = SD.open("gps_data.txt", FILE_WRITE);
  if (gpsFile) {
      gpsFile.println("START OF NEW GPS RECORDING");
      gpsFile.println("Time,Latitude,Longitude,Altitude,SOG,COG,linear acceleration x,y,z,magnetic x,y,z,gyro x,y,z,");
      gpsFile.close();
  } else {
      Serial.println("Error opening gps_data.txt");
  }

  imuFile = SD.open("gps_data.txt", FILE_WRITE);
  if (imuFile) {
      // imuFile.println("START OF NEW IMU RECORDING");
      imuFile.close();
  } else {
      Serial.println("Error opening gps_data.txt");
  }
}

uint32_t timer = millis();
void loop()
{
  // GPS part
  char c = GPS.read();
  // if you want to debug, this is a good time to do it!
  if ((c) && (GPSECHO))
    Serial.write(c);

  // if a sentence is received, we can check the checksum, parse it...
  if (GPS.newNMEAreceived()) {
  // a tricky thing here is if we print the NMEA sentence, or data
  // we end up not listening and catching other sentences!
  // so be very wary if using OUTPUT_ALLDATA and trytng to print out data
  //Serial.println(GPS.lastNMEA());   // this also sets the newNMEAreceived() flag to false

    if (!GPS.parse(GPS.lastNMEA()))   // this also sets the newNMEAreceived() flag to false
      return;  // we can fail to parse a sentence in which case we should just wait for another
  }
   
  if (millis() - timer > 200) {
    timer = millis(); //reset the timer
    Serial.print("Fix: "); Serial.print((int)GPS.fix);
    Serial.print(" quality: "); Serial.println((int)GPS.fixquality);

    // gpsFile = SD.open("gps_data.txt", FILE_WRITE);
    // if (gpsFile) {
    //   gpsFile.print("\nFix: "); gpsFile.print((int)GPS.fix);
    //   gpsFile.print(" quality: "); gpsFile.println((int)GPS.fixquality);
    //   gpsFile.close();
    // } 

    if(GPS.fix)
    {
      gpsFile = SD.open("gps_data.txt", FILE_WRITE);
      if (gpsFile) {
        Serial.println("GPS should be saving now");
        // gpsFile.print("\nTime: ");
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
        gpsFile.print(GPS.milliseconds);
        gpsFile.print(",");
        // gpsFile.print("\nFix: "); gpsFile.print((int)GPS.fix);
        // gpsFile.print(" quality: "); gpsFile.println((int)GPS.fixquality);
  
        // gpsFile.print("Location: ");
        gpsFile.print(GPS.latitude, 4); gpsFile.print(GPS.lat);
        gpsFile.print(",");
        gpsFile.print(GPS.longitude, 4); gpsFile.print(GPS.lon);
        gpsFile.print(",");
        gpsFile.print(GPS.altitude);
        gpsFile.print(",");
        gpsFile.print(GPS.speed);
        gpsFile.print(",");
        gpsFile.print(GPS.angle);
        gpsFile.print(",");
      
        gpsFile.close();
      } else {
          Serial.println("Error opening gps file!");
      }

      imuFile = SD.open("gps_data.txt", FILE_WRITE);
      if (imuFile) {
        
        lsm.read();  /* ask it to read in the data */ 
        /* Get a new sensor event */ 
        sensors_event_t a, m, g, temp;
        lsm.getEvent(&a, &m, &g, &temp); 

        Serial.println("IMU should be saving now");
        // imuFile.print("Acceleration: ");
        imuFile.print(a.acceleration.x); imuFile.print(",");
        imuFile.print(a.acceleration.y); imuFile.print(",");
        imuFile.print(a.acceleration.z); imuFile.print(",");
        // Magnetic:
        imuFile.print(m.magnetic.x); imuFile.print(",");
        imuFile.print(m.magnetic.y); imuFile.print(",");
        imuFile.print(m.magnetic.z); imuFile.print(",");
        // imuFile.print(Gyro: "");
        imuFile.print(g.gyro.x); imuFile.print(",");
        imuFile.print(g.gyro.y); imuFile.print(",");
        imuFile.print(g.gyro.z); imuFile.print("\n");

        imuFile.close();
      } else {
          Serial.println("Error opening imu file!");
      }
    }
  }
}