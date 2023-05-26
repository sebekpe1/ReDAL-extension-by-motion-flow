import customtkinter
import tkinter as tk
from RangeSlider.RangeSlider import RangeSliderH
import open3d as o3d
import numpy as np
import copy


class App(customtkinter.CTk):
    def __init__(self, classes, min_val, max_val, pcl, labels, colors=True, config=None, name=f'point_cloud',
                 background_points=None, background_labels=None):
        super().__init__()

        self.pcd_pcl = pcl
        self.pcd_labels = labels
        self.pcd_colors = colors
        self.pcd_config = config
        self.pcd_name = name
        self.pcd_background_points = background_points
        self.pcd_background_labels = background_labels
        self.vis_combine = o3d.visualization.Visualizer()
        self.vis_combine.create_window(window_name='Combine visualization')
        self.vis_single = o3d.visualization.Visualizer()
        self.vis_single.create_window(window_name='Single visualization', width=500, height=750)
        self.combine_pcl = pcl[0]
        max_points = (0, 0)
        for i in range(1, len(pcl)):
            self.combine_pcl = np.vstack((self.combine_pcl, pcl[i]))
            if pcl[i].shape[0] > max_points[0]:
                max_points = (pcl[i].shape[0], i)
        self.cloud = o3d.geometry.PointCloud()
        self.pcd_single = o3d.geometry.PointCloud()

        self.geometry("500x750")
        self.title("Annotators interface")

        # add widgets to app
        self.frame_1 = customtkinter.CTkFrame(master=self)
        self.frame_1.pack(pady=20, padx=60, fill="both", expand=True)

        self.label_1 = customtkinter.CTkLabel(master=self.frame_1, text='Time frames selection', justify=customtkinter.LEFT)
        self.label_1.pack(pady=10, padx=10)

        self.slider_low = tk.IntVar()
        self.slider_hight = tk.IntVar()
        self.range_slider = RangeSliderH(self.frame_1, [self.slider_low, self.slider_hight], Width=300, padX=17, min_val=min_val, max_val=max_val,
                                    show_value=False, line_s_color='#7eb1c2')
        self.range_slider.pack(pady=10, padx=10)

        self.slider_low.trace_add('write', self.print_range_sdlider_value)
        self.slider_hight.trace_add('write', self.print_range_sdlider_value)

        self.lower_value = 0
        self.higher_value = 1

        self.button_combine = customtkinter.CTkButton(master=self.frame_1, text='Combine visualization control',
                                                command=self.run_vis_combine)
        self.button_combine.pack(pady=10, padx=10)

        self.button_single = customtkinter.CTkButton(master=self.frame_1, text='Single visualization control',
                                                command=self.run_vis_single)
        self.button_single.pack(pady=10, padx=10)

        self.optionmenu_1 = customtkinter.CTkOptionMenu(self.frame_1,
                                                   values=classes)
        self.optionmenu_1.pack(pady=10, padx=10)
        self.optionmenu_1.set("Label selection")

        self.button_1 = customtkinter.CTkButton(master=self.frame_1, text='Label', fg_color='#008000', command=self.label_button_callback)
        self.button_1.pack(pady=10, padx=10)

        self.button_2 = customtkinter.CTkButton(master=self.frame_1, text='Delete', fg_color='#d2042d', command=self.delete_button_callback)
        self.button_2.pack(pady=10, padx=10)

        self.update_pcd_vis()

        self.output_label = None

    def start(self):
        self.after(1000, self.update_pcd_vis)
        self.mainloop()
        return (self.lower_value, self.higher_value), self.output_label

    def update_pcd_vis(self):
        self.after(1000, self.update_pcd_vis)
        self.point_cloud_visualization()
    # add methods to app
    def button_click(self):
        print("button click")

    def label_button_callback(self):

        if self.optionmenu_1._current_value not in ['Label selection', 'Select label']:
            self.output_label = self.optionmenu_1._current_value
            self.destroy()
        else:
            self.optionmenu_1.set("Select label")

    def delete_button_callback(self):
        self.destroy()

    def run_vis_combine(self):
        self.vis_combine.run()

    def run_vis_single(self):
        self.vis_single.run()

    def print_range_sdlider_value(self, var, index, mode):
        pass

    def point_cloud_visualization(self):
        max_points = (0, 0)
        instance_rgb = None

        self.lower_value = copy.deepcopy(self.slider_low.get())
        self.higher_value = copy.deepcopy(self.slider_hight.get())

        for i in range(len(self.pcd_pcl)):
            frame_rgb = np.zeros((self.pcd_pcl[i].shape[0], 3))
            if self.lower_value <= i <= self.higher_value:
                frame_rgb[:, 1] = 1
                if self.pcd_pcl[i].shape[0] > max_points[0]:
                    max_points = (self.pcd_pcl[i].shape[0], i)
            else:
                frame_rgb[:, 0] = 1
            if instance_rgb is None:
                instance_rgb = frame_rgb
            else:
                instance_rgb = np.vstack((instance_rgb, frame_rgb))

        instance_xyz = self.combine_pcl
        all_xyz = np.vstack((instance_xyz, self.pcd_background_points))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_xyz)
        if self.pcd_colors:
            background_rgb = np.zeros((self.pcd_background_labels.shape[0], 3))
            all_rgb = np.vstack((instance_rgb, background_rgb))
            pcd.colors = o3d.utility.Vector3dVector(all_rgb)

        self.cloud = pcd

        self.pcd_single = o3d.geometry.PointCloud()
        self.pcd_single.points = o3d.utility.Vector3dVector(self.pcd_pcl[max_points[1]] - np.mean(self.pcd_pcl[self.slider_low.get()] + np.array([10, 0, 0]), axis=0))
        self.pcd_single.colors = o3d.utility.Vector3dVector(np.zeros((self.pcd_pcl[max_points[1]].shape[0], 3)))

        self.vis_combine.remove_geometry(self.cloud)
        self.vis_combine.clear_geometries()
        self.vis_combine.add_geometry(self.cloud)
        self.vis_combine.poll_events()
        self.vis_combine.update_renderer()

        self.vis_single.remove_geometry(self.pcd_single)
        self.vis_single.clear_geometries()
        self.vis_single.add_geometry(self.pcd_single)
        self.vis_single.reset_view_point(True)
        self.vis_single.poll_events()
        self.vis_single.update_renderer()

if __name__ == '__main__':
    app = App(['Car', 'Pedestrian', 'Bus'], 0, 100)
    app.start()

    for i in range(100):
        print(i)
