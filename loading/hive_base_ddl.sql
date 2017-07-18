DROP TABLE qc_results;
CREATE EXTERNAL TABLE qc_results(
product_id string,
qc_lot_number string,
run_date string,
x_value int,
y_value int,
z_value int
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES(
"separatorChar" = ",",
"quoteChar" = "'",
"escapeChar" = '\\'
)
STORED AS TEXTFILE
LOCATION '/user/w205/final';



DROP TABLE flagging;
CREATE EXTERNAL TABLE flagging(
product_id string,
run_date string,
IPU_version string,
count int,
counter_name string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES(
"separatorChar" = ",",
"quoteChar" = "'",
"escapeChar" = '\\'
)
STORED AS TEXTFILE
LOCATION '/user/w205/final';



DROP TABLE xbarm;
CREATE EXTERNAL TABLE xbarm
(
product_id string,
IPU_version string,
x_xbarm int,
y_xbarm int,
z_xbarm int
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES(
"separatorChar" = ",",
"quoteChar" = "'",
"escapeChar" = '\\'
)
STORED AS TEXTFILE
LOCATION '/user/w205/final';



DROP TABLE reference_set;
CREATE EXTERNAL TABLE reference_set(
product_id string,
serial_number string,
model_id string,
IPU_version string,
run_date string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES(
"separatorChar" = ",",
"quoteChar" = "'",
"escapeChar" = '\\'
)
STORED AS TEXTFILE
LOCATION '/user/w205/final';



DROP TABLE lot_dates;
CREATE EXTERNAL TABLE lot_dates(
QC_lot_number string,
start_date date,
end_date date,
begin_use_date date
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES(
"separatorChar" = ",",
"quoteChar" = "'",
"escapeChar" = '\\'
)
STORED AS TEXTFILE
LOCATION '/user/w205/final';
