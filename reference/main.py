import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure, output_file, show

q_data = pd.read_csv('C:/Users/yangyq/Google Drive/Grad school project/ucberkeley_mids/academics/W205_Storage/finalprojmisc/Specific QC Lot Number Results.csv')
q_data.columns =['product_id', 'qc_lot', 'time', 'w-x', 'w-y', 'w-z']
q_data.dropna()

q_data.head()

my_data = dict(x=[], y1=[], y2=[], y3=[])
source = ColumnDataSource(my_data)

fig = Figure()
fig.line(source=source, x="x", y="y1", line_width=2, alpha=.85, color="red")
fig.line(source=source, x="x", y="y2", line_width=2, alpha=.85, color="blue")
fig.line(source=source, x="x", y="y3", line_width=2, alpha=.85, color="green")

ct = 0
w_x = 0
w_y = 0
w_z = 0
def update_data():
    global ct, w_x, w_y, w_z
    ct += 1
    temp_rec = q_data.iloc[[ct]]
    w_x = temp_rec["w-x"]
    w_y = temp_rec["w-y"]
    w_z = temp_rec["w-z"]
    new_data = dict(x=[ct], y1=[w_x], y2=[w_y], y3=[w_z])
    source.stream(new_data, 100)

curdoc().add_root(fig)
curdoc().add_periodic_callback(update_data, 100)