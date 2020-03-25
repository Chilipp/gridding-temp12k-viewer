import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML, clear_output
import cartopy.crs as ccrs
import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import ipywidgets as widgets

class ScatterplotWidget(object):

    def __init__(self, df, scatter_ax, map_ax, alpha_other=0.3, selection_color=[1., 0., 0., 1.],
                 selection_size=3):
        self.initial_df = df.set_index(['lon', 'lat'])
        self.df = df.reset_index().drop_duplicates(['lon', 'lat', 'time']).set_index(['lon', 'lat'])

        collection = scatter_ax.scatter(self.df.time, self.df.temperature, 1)

        self.lola = lola = df.drop_duplicates(['lon', 'lat']).reset_index().set_index(['lon', 'lat'])
        lola['index'] = np.arange(len(lola))
        map_collection = map_ax.scatter(
            lola.index.get_level_values(0),
            lola.index.get_level_values(1), s=1, transform=ccrs.PlateCarree())

        self.table_widget = widgets.Output()

        self.canvas = scatter_ax.figure.canvas
        self.collection = collection
        self.map_collection = map_collection

        self.selection_color = selection_color
        self.alpha_other = alpha_other
        self.selection_size = selection_size

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.map_xys = map_collection.get_offsets()
        self.map_Npts = len(self.map_xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        # Ensure that we have separate colors for each object
        self.map_fc = map_collection.get_facecolors()
        if len(self.map_fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.map_fc) == 1:
            self.map_fc = np.tile(self.map_fc, (self.map_Npts, 1))

        self.lasso = LassoSelector(scatter_ax, onselect=self.onselect)
        self.map_lasso = LassoSelector(map_ax, onselect=self.map_onselect)
        self.ind = []
        self.map_ind = []
        self.refresh_table()
        self.create_buttons()
        display(self.table_widget)

    def onselect(self, verts):
        path = Path(verts)
        self.update_points(np.nonzero(path.contains_points(self.xys))[0])

        selected = self.df.iloc[self.ind]
        self.update_map(
            self.lola.loc[selected.index, 'index'].values)

        self.refresh_table()

        self.canvas.draw_idle()

    def update_points(self, ind):
        if not len(self.ind) or not self.select_intersection:
            self.ind = ind
        else:
            self.ind = ind[np.isin(ind, self.ind)]
        if self.select_whole_series:
            s = pd.Series(np.arange(len(self.df)), index=self.df.index)
            self.ind = s.loc[np.unique(self.df.iloc[self.ind].index)].values
        fc = self.fc.copy()
        fc[:, -1] = self.alpha_other
        fc[self.ind, :] = [self.selection_color]
        self.collection.set_facecolors(fc)
        sizes = np.ones(len(fc))
        sizes[self.ind] = self.selection_size
        self.collection.set_sizes(sizes)

    def map_onselect(self, verts):
        path = Path(verts)
        self.update_map(np.nonzero(path.contains_points(self.map_xys))[0])

        selected = self.lola.iloc[self.map_ind]
        self.update_points(
            self.df.loc[selected.index, 'index'].values)

        self.refresh_table()

        self.canvas.draw_idle()

    @property
    def select_whole_series(self):
        return self.btn_whole_line.value

    @property
    def select_intersection(self):
        return self.btn_intersect.value

    def create_buttons(self):
        self.btn_whole_line = widgets.ToggleButton(
            description='Whole series',
            tooltip='Select the whole time-series for the selected points in the time-age-scatter plot'
        )

        self.btn_intersect = widgets.ToggleButton(
            description="Intersection",
            tooltip="Keep only the intersection with the current selection")

        self.btn_reset = widgets.Button(
            description="Reset",
            tooltip="Clear the selection")
        self.btn_reset.on_click(self.clear_selection)

        self.btn_export_selection = widgets.Button(
            description="Export to",
            tooltip="Export the selection to an Excel file"
        )
        self.btn_export_selection.on_click(self.export_selection)

        self.txt_export = widgets.Text(
            value='selection.xlsx',
            placeholder='Enter a filename',
        )

        self.out_download = widgets.Output()

        display(
            widgets.VBox([
                widgets.HBox([self.btn_whole_line, self.btn_intersect, self.btn_reset]),
                widgets.HBox([self.btn_export_selection, self.txt_export, self.out_download])
            ]))

    def export_selection(self, *args, **kwargs):
        target_file = self.txt_export.value
        if not len(self.ind):
            selection = self.initial_df
        else:
            selection = self.initial_df.loc[self.df.iloc[self.ind].index.drop_duplicates()]
        selection.to_excel(target_file)
        self.out_download.clear_output()
        self.out_download.clear_output()
        self.out_download.append_display_data(HTML(
            f"<a href='{target_file}' target='_blank' "
            f" title='Download the selection'>"
            f"     Download {target_file}"
            f"</a>"))

    def clear_selection(self, *args, **kwargs):
        ind = np.array([], dtype=int)
        self.update_points(ind)
        self.update_map(ind)
        self.collection.set_facecolors(self.fc)
        self.map_collection.set_facecolors(self.map_fc)
        self.refresh_table()

    def refresh_table(self):
        self.table_widget.clear_output()
        self.table_widget.clear_output()
        ind = np.unique(self.map_ind)
        selected = self.lola.iloc[
            ind if len(ind) else slice(None)]
        self.table_widget.append_display_data(HTML(
            selected.to_html(
                max_rows=50, escape=False)))

    def update_map(self, ind):
        if not len(self.ind) or not self.select_intersection:
            self.map_ind = ind
        else:
            self.map_ind = ind[np.isin(ind, self.map_ind)]
        fc = self.map_fc.copy()
        fc[:, -1] = self.alpha_other
        fc[self.map_ind, :] = [self.selection_color]
        self.map_collection.set_facecolors(fc)
        sizes = np.ones(len(fc))
        sizes[self.map_ind] = self.selection_size
        self.map_collection.set_sizes(sizes)

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.map_lasso.disconnect_events()
        self.map_fc[:, -1] = 1
        self.collection.set_facecolors(self.map_fc)
        self.canvas.draw_idle()
