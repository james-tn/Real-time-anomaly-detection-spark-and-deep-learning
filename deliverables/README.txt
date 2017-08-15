=====================================================
Project Dependencies
=====================================================
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




=====================================================
EC2 Instance file structure
=====================================================
/home/ubuntu
    certs
        mycert.pem
    data
        Specific QC Lot Number Results.csv
        XBarM Result Set for Steve (CSV Form Revised).csv
        Reference Set A Result Set.csv
        Test Flagging Query Data Set.csv
    Notebooks
        w205final_v1.0.ipynb
        main.py
        sysmex_anomaly_model1.index
        sysmex_anomaly_model1.meta


        
=====================================================
Project Github file structure
=====================================================
https://github.com/james-tn/w205final
    /deliverables
        README.txt
        w205_section04_final_project_james_weixing_annalaissa_yang.pdf
        w205_section04_final_project_james_weixing_annalaissa_yang.pptx
        w205final_v1.0.ipynb
        main.py
        qc_over_time.html
        wx_select.html
        wx_wy.html
        QC_plots_Weixing.png
    /reference
        
        
        
        
=====================================================
How to run the pieces of the project
=====================================================
        

        

A) How to run the data transformation and build the model

... On EC2 instance
1) Ensure the data files are located in the /home/ubuntu/data folder
    Specific QC Lot Number Results.csv
    XBarM Result Set for Steve (CSV Form Revised).csv
    Reference Set A Result Set.csv
    Test Flagging Query Data Set.csv

2) Run the jupyter notebook located in the /home/ubuntu/Notebooks folder
    w205final_v1.0.ipynb
    
    This should:
        i) create the tensforflow model files that will be used by bokeh
            /home/ubuntu/Notebooks/sysmex_anomaly_model1.index
            /home/ubuntu/Notebooks/sysmex_anomaly_model1.meta
        
        ii) load transformed data into a localhost MySQL database
            Database = analyzer
            Tables = qc_data, x_bar_m
    
    
    

B) How to run the Bokeh dashboard

... On EC2 instance
1) Ensure the at least the following data files are located in the /home/ubuntu/data folder
    Specific QC Lot Number Results.csv
    XBarM Result Set for Steve (CSV Form Revised).csv


2) Ensure the tensorflow model files are in the /home/ubuntu/Notebooks/ folder
    sysmex_anomaly_model1.meta
    sysmex_anomaly_model1.index


3) start the bokeh server with the following command
    bokeh serve --show main.py --allow-websocket-origin=ec2-54-204-252-120.compute-1.amazonaws.com:5006


... On Workstation
4) connect to the dashboard from your workstation's browser
    
    for streaming dashboard
    http://ec2-54-204-252-120.compute-1.amazonaws.com:5006/main
    
    for interactive visualizations, in "deliverables" folder
    qc_over_time.html
    wx_select.html
    wx_wy.html
    


5) Might need to wait for the main.py code to start




C) How to connect with tableau
1) Open the tableau workbook located in "deliverables" folder

2) Is already be set up to use qc_data.tde, and x_bar_m.tde
    Note, if connection is broken please ensure you are connecting with the following credentials
    Type = MySQL
    Server = ec2-54-204-252-120.compute-1.amazonaws.com
    Port = 3306
    Username = analyzer_user
    Pass = abc123
    Database = analyzer
    Tables = qc_data, x_bar_m
    
    
    
    
D) How to run the jupyter server
    assuming cert exists
        /home/ubuntu/certs/mycert.pem
    and config files exist
        /home/ubuntu/.jupyter/jupyter_notebook_config.py
        
    See details in github file
        ../reference/Multiple Remote Jupyter Servers.txt
        