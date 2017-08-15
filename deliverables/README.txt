Project Dependencies

Shared EC2 instance, with the following installed
* Spark 
* MySQL
* Anaconda, including
    * Bokeh
    * SQLAlchemy

Workstation with the following installed
* Tableau
* Tableau MySQL driver
* Tableau Mac driver

    
    
How to run the Bokeh dashboard

... On EC2 instance
1) Ensure the data files are located in the /home/ubuntu/data folder
Specific QC Lot Number Results.csv
XBarM Result Set for Steve (CSV Form Revised).csv


2) Ensure the tensorflow model files are in the /home/ubuntu/Notebooks/ folder
sysmex_anomaly_model1.meta
sysmex_anomaly_model1.index


3) start the bokeh server with the following command
bokeh serve --show main.py --allow-websocket-origin=ec2-54-204-252-120.compute-1.amazonaws.com:5006


... On Workstation
4) connect to the dashboard from your workstation's browser
http://ec2-54-204-252-120.compute-1.amazonaws.com:5006/main


5) Might need to wait for the main.py code to start




How to connect with tableau
1) Open the tableau workbook located in "deliverables" folder

2) Is already be set up to use qc_data.tde, and x_bar_m.tde

Note, if connection is broken please ensure you are connecting with the following credentials
Type = MySQL
Server = ec2-54-204-252-120.compute-1.amazonaws.com
Port = 3306
Username = analyzer_user
Pass = abc123
Tables = qc_data, x_bar_m