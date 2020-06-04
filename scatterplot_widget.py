import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML, clear_output
import cartopy.crs as ccrs
import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import ipywidgets as widgets
import qgrid

class ScatterplotWidget(object):

    def __init__(self, df, scatter_ax, map_ax, alpha_other=0.3, selection_color=[1., 0., 0., 1.],
                 selection_size=3):

        self.canvas = scatter_ax.figure.canvas
        self.selection_color = selection_color
        self.alpha_other = alpha_other
        self.selection_size = selection_size
        self.scatter_ax = scatter_ax
        self.map_ax = map_ax

        self.create_buttons()
        self.table_widget = qgrid.show_grid(
            df, show_toolbar=False,
            grid_options={'editable': False, 'forceFitColumns': False})

        self.init_df(df)

    def init_df(self, df):

        self.exclude = set()
        self.mean_line = None

        self.initial_df = df.set_index(['lon', 'lat'])
        self.df = df.reset_index().drop_duplicates(['lon', 'lat', 'time']).set_index(['lon', 'lat'])

        collection = self.scatter_ax.scatter(
            self.df.time, self.df.temperature, 1)

        self.lola = lola = df.drop_duplicates(['lon', 'lat']).reset_index().set_index(['lon', 'lat'])
        lola['index'] = np.arange(len(lola))
        map_collection = self.map_ax.scatter(
            lola.index.get_level_values(0),
            lola.index.get_level_values(1), s=1, transform=ccrs.PlateCarree())

        self.collection = collection
        self.map_collection = map_collection

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

        self.lasso = LassoSelector(self.scatter_ax, onselect=self.onselect)
        self.map_lasso = LassoSelector(self.map_ax, onselect=self.map_onselect)
        self.plot_band_mean()
        self.ind = []
        self.map_ind = []
        self.refresh_table()

    @property
    def band_mean(self):
        df = self.initial_df
        if self.exclude:
            df = df[~df.TSid.isin(self.exclude)]
        cell_means = df.groupby(
            ['clon', 'clat', 'time']).temperature.mean()
        band_mean = cell_means.groupby('time').mean()
        return band_mean

    def plot_band_mean(self):
        band_mean = self.band_mean
        if self.mean_line is not None:
            self.mean_line.remove()
        self.mean_line = self.collection.axes.plot(
            band_mean.index, band_mean.values, c='magenta', lw=4)[0]

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

    @property
    def show_all_samples(self):
        return self.btn_show_all_rows.value

    def create_buttons(self):
        self.btn_whole_line = widgets.ToggleButton(
            description='Whole series',
            tooltip='Select the whole time-series for the selected points in the time-age-scatter plot'
        )

        self.btn_intersect = widgets.ToggleButton(
            description="Intersection",
            tooltip="Keep only the intersection with the current selection")

        self.btn_show_all_rows = widgets.ToggleButton(
            description="Show all rows",
            tooltip="Show all rows of the corresponding grid cell")

        self.btn_exclude_selection = widgets.Button(
            description="Exclude in mean",
            tooltip=("Exclude the selected time-series in the calculation of "
                     "the band mean"))
        self.btn_exclude_selection.on_click(self.exclude_selection)

        self.btn_include_selection = widgets.Button(
            description="Include in mean",
            tooltip=("Include the selected time-series (or all if nothing is "
                     "selected) in the calculation of the band mean"))
        self.btn_include_selection.on_click(self.include_selection)

        self.btn_reset = widgets.Button(
            description="Reset",
            tooltip="Clear the selection")
        self.btn_reset.on_click(self.clear_selection)

        self.btn_refresh_table = widgets.Button(
            description="Refresh table",
            tooltip="Refresh the table with the current selection")
        self.btn_refresh_table.on_click(self.refresh_table)

        self.btn_export_selection = widgets.Button(
            description="Export to",
            tooltip="Export the selection to an Excel file"
        )
        self.btn_export_selection.on_click(self.export_selection)

        self.txt_export = widgets.Text(
            value='selection.xlsx',
            placeholder='Enter a filename',
        )

        self.btn_copy_ids = widgets.Button(
            description="Copy IDs",
            tooltip="Copy the selected TSids to clipboard")
        self.btn_copy_ids.on_click(self.copy_ids)

        self.btn_copy_datasets = widgets.Button(
            description="Copy datasetNames",
            tooltip="Copy selected LiPD datasets names to clipboard")
        self.btn_copy_datasets.on_click(self.copy_dataset_names)

        self.btn_copy_table = widgets.Button(
            description="Copy table",
            tooltip="Copy the entire table to clipboard")
        self.btn_copy_table.on_click(self.copy_all)

        self.out_download = widgets.Output()

        self.vbox = widgets.VBox([
                widgets.HBox([self.btn_whole_line, self.btn_intersect]),
                widgets.HBox([self.btn_exclude_selection,
                              self.btn_include_selection]),
                widgets.HBox([self.btn_reset, self.btn_refresh_table,
                              self.btn_show_all_rows]),
                widgets.HBox([self.btn_export_selection, self.txt_export,
                              self.out_download]),
                widgets.HBox([self.btn_copy_ids, self.btn_copy_datasets,
                              self.btn_copy_table]),
        ])

    def include_selection(self, *args, **kwargs):
        if not len(self.ind):
            self.exclude.clear()
            self.plot_band_mean()
        else:
            self.exclude.difference_update(self.df.iloc[self.ind].TSid)
            self.plot_band_mean()
        self.canvas.draw_idle()

    def exclude_selection(self, *args, **kwargs):
        if not len(self.ind):
            pass
        else:
            self.exclude.update(self.df.iloc[self.ind].TSid)
            self.plot_band_mean()
            self.canvas.draw_idle()

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

    @property
    def selection(self):
        return self.table_widget.get_changed_df()

    def copy_ids(self, *args, **kwargs):
        self.copy_columns('TSid')

    def copy_dataset_names(self, *args, **kwargs):
        self.copy_columns('dataSetName')

    def copy_all(self, *args, **kwargs):
        self.selection.to_clipboard(index=False, header=True)

    def copy_columns(self, *cols):
        cols = list(cols)
        selection = self.selection.drop_duplicates(cols)[cols]
        selection.to_clipboard(index=False, header=False)

    def clear_selection(self, *args, **kwargs):
        ind = np.array([], dtype=int)
        self.update_points(ind)
        self.update_map(ind)
        self.collection.set_facecolors(self.fc)
        self.map_collection.set_facecolors(self.map_fc)
        self.refresh_table()

    def refresh_table(self, *args, **kwargs):
        ind = np.unique(self.map_ind)
        selected = self.lola.iloc[
            ind if len(ind) else slice(None)]
        if self.show_all_samples:
            selected = self.initial_df.loc[selected.index]
        self.table_widget.df = selected

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
