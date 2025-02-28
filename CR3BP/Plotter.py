import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np

class Plotter:

    def __init__(self, exact, pred, radius, alt, e, num_orbits, planet, plot_training_data, fig_handle):
        self.exact = exact
        self.pred = pred
        self.radius = radius
        self.alt = alt
        self.e = e
        self.num_orbits = num_orbits
        self.planet = planet
        self.plot_training_data = plot_training_data
        self.fig_handle = fig_handle
        self.colours = ['black', 'red','deepskyblue', 'orange', 'pink', 'silver']
        self.line = ['-', '--',':']
        self.titles = ['X Position', 'Y Position', 'X Velocity', 'Y Velocity']

    def perifocalPlot(self):
        training_lim = torch.load("Training_Plot.pt")
        # Setting up Spherical Earth to Plot
        N = 50
        phi = np.linspace(0, 2 * np.pi, N)
        theta = np.linspace(0, np.pi, N)
        theta, phi = np.meshgrid(theta, phi)
        X_Earth = self.radius * np.cos(phi) * np.sin(theta)
        Y_Earth = self.radius * np.sin(phi) * np.sin(theta)

        # Plotting Earth and Orbit
        mpl.rc('font',family='Serif')
        fig = plt.figure(figsize = (8,8))
        ax = plt.axes()
        if self.planet == 'Earth':
            planetColor = 'midnightblue'
            if self.plot_training_data:
                ax.plot(training_lim[:,0,0], training_lim[:,1,0], 'green')
                ax.plot(training_lim[:,0,1], training_lim[:,1,1], 'green')
                ax.fill(training_lim[:,0,1], training_lim[:,1,1], alpha=0.2, facecolor='green', label='Training Data')
                ax.fill(training_lim[:,0,0], training_lim[:,1,0],facecolor='white')
        elif self.planet == 'Moon':
            planetColor = 'grey'
        elif self.planet == 'Jupiter':
            planetColor = 'burlywood'
        ax.fill(X_Earth, Y_Earth, color=planetColor)
        # plt.title('Two-Body Orbit')
        ax.set_xlabel('X [km]', fontsize = 16)
        ax.set_ylabel('Y [km]', fontsize = 16)

        for j in range(len(self.exact)):
            exactPlot = self.exact[j]
            predPlot = self.pred[j].detach().cpu().numpy()
            ax.plot(exactPlot[0,:], exactPlot[1,:], 'black', label='Exact Solution')
            ax.plot(predPlot[0,:], predPlot[1,:], linestyle='--', color=self.colours[j+1], label=f'e = {self.e[j]}, alt = {self.alt[j]} km')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # ax.set_title(f'Predicted solution for orbits of varying radii - {self.planet}', fontsize = 16)
        ax.set_aspect('equal')
        plt.savefig(f"perifocal-{self.fig_handle}.svg", format="svg", dpi=1200)
        plt.show()
        

    def statePlot(self):
        fig, ax = plt.subplots(2, 2, sharex = 'col', sharey = 'row', figsize = (12, 8))
        fig.tight_layout(pad = 3.0)

        for j in range(len(self.exact)):
            exactPlot = self.exact[j]
            predPlot = self.pred[j].detach().cpu().numpy()
            for i in range(4):
                if i == 0:
                    a = 0
                    b = 0
                    ylabel = "Position [km]"
                elif i == 1:
                    a = 0
                    b = 1
                    ylabel = "Position [km]"
                elif i == 2:
                    a = 1
                    b = 0
                    ylabel = "Velocity [km/s]"
                elif i == 3:
                    a = 1
                    b = 1
                    ylabel = "Velocity [km/s]"
                ax[a,b].plot(exactPlot[i,:], color=self.colours[0], linestyle=self.line[0], label='_nolegend_', linewidth=2)
                ax[a,b].plot(predPlot[i,:], color=self.colours[j+1], linestyle=self.line[1], label=f'e = {self.e[j]}, alt = {self.alt[j]} km', linewidth=2)
                ax[a,b].set_title(self.titles[i], fontsize = 16)
                ax[a,b].set_ylabel(ylabel, fontsize = 16)

        ax[1,0].set_xlabel("Number of Orbits", fontsize = 16)
        ax[1,1].set_xlabel("Number of Orbits", fontsize = 16)
        ax[1,0].set_xticks([0, self.num_orbits*80, self.num_orbits*160, self.num_orbits*240, self.num_orbits*320, self.num_orbits*400, self.num_orbits*480, self.num_orbits*560, self.num_orbits*640, self.num_orbits*720, self.num_orbits*800],[0, self.num_orbits*0.1, self.num_orbits*0.2, self.num_orbits*0.3, self.num_orbits*0.4, self.num_orbits*0.5, self.num_orbits*0.6, self.num_orbits*0.7, self.num_orbits*0.8, self.num_orbits*0.9, self.num_orbits*1.00])
        ax[1,1].set_xticks([0, self.num_orbits*80, self.num_orbits*160, self.num_orbits*240, self.num_orbits*320, self.num_orbits*400, self.num_orbits*480, self.num_orbits*560, self.num_orbits*640, self.num_orbits*720, self.num_orbits*800],[0, self.num_orbits*0.1, self.num_orbits*0.2, self.num_orbits*0.3, self.num_orbits*0.4, self.num_orbits*0.5, self.num_orbits*0.6, self.num_orbits*0.7, self.num_orbits*0.8, self.num_orbits*0.9, self.num_orbits*1.00])
        # fig.suptitle(f'Two-Body Problem States - {self.planet}', fontsize = 18)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0,1].legend(by_label.values(), by_label.keys())
        plt.savefig(f"states-{self.fig_handle}.svg", format="svg", dpi=1200)


    def errorPlot(self, errorType):
        # Setting up Error plot
        fig, ax = plt.subplots(2,1, sharey = 'row', figsize = (10,8))
        fig.tight_layout(pad = 3.0)

        for j in range(len(self.exact)):
            exactPlot = self.exact[j]
            predPlot = self.pred[j].detach().cpu().numpy()
            ax[0].plot(exactPlot[0,:] - predPlot[0,:], color=self.colours[j+1], label= '_nolegend_')
            ax[1].plot(exactPlot[1,:] - predPlot[1,:], color=self.colours[j+1], label = f'e = {self.e[j]}, alt = {self.alt[j]} km')
            ax[0].set_ylabel('Error in X [km]', fontsize = 16)
            ax[1].set_ylabel('Error in Y [km]', fontsize = 16)
            ax[1].set_xlabel('Number of Orbits', fontsize = 16)
        ax[0].set_xticks([0, self.num_orbits*80, self.num_orbits*160, self.num_orbits*240, self.num_orbits*320, self.num_orbits*400, self.num_orbits*480, self.num_orbits*560, self.num_orbits*640, self.num_orbits*720, self.num_orbits*800],[0, self.num_orbits*0.1, self.num_orbits*0.2, self.num_orbits*0.3, self.num_orbits*0.4, self.num_orbits*0.5, self.num_orbits*0.6, self.num_orbits*0.7, self.num_orbits*0.8, self.num_orbits*0.9, self.num_orbits*1.00])
        ax[1].set_xticks([0, self.num_orbits*80, self.num_orbits*160, self.num_orbits*240, self.num_orbits*320, self.num_orbits*400, self.num_orbits*480, self.num_orbits*560, self.num_orbits*640, self.num_orbits*720, self.num_orbits*800],[0, self.num_orbits*0.1, self.num_orbits*0.2, self.num_orbits*0.3, self.num_orbits*0.4, self.num_orbits*0.5, self.num_orbits*0.6, self.num_orbits*0.7, self.num_orbits*0.8, self.num_orbits*0.9, self.num_orbits*1.00])
        fig.suptitle(f'{errorType} Error in Position States - {self.planet}', fontsize = 18)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys())
        plt.savefig(f"{errorType}error-{self.fig_handle}.svg", format="svg", dpi=1200)

    def metricPlot(self):
        fig, ax = plt.subplots(2,2, sharex = 'col', figsize = (15, 10))
        fig.tight_layout(pad = 3.0)

        for j in range(len(self.exact)):
            exactPlot = self.exact[j]
            predPlot = self.pred[j].detach().cpu().numpy()

            r = (predPlot[0,:]**2 + predPlot[1,:]**2)**0.5
            r_avg =np.mean(r)
            r_metric = r/r_avg - 1

            v = (predPlot[2,:]**2 + predPlot[3,:]**2)**0.5
            v_avg = np.mean(v)
            v_metric = v/v_avg - 1

            rdotv = predPlot[0,:]*predPlot[2,:] + predPlot[1,:]*predPlot[3,:]
            rdotv_avg = np.mean(rdotv)
            rdotv_metric = rdotv/rdotv_avg - 1

            Lz = predPlot[0,:]*predPlot[3,:] - predPlot[1,:]*predPlot[2,:]
            Lz_avg = np.mean(Lz)
            Lz_metric = Lz/Lz_avg - 1

            ax[0,0].plot(r_metric, label='_nolegend_', color=self.colours[j+1], linewidth=2)
            ax[0,1].plot(v_metric, label = '_nolegend_',color=self.colours[j+1], linewidth=2)
            ax[1,0].plot(rdotv_metric, label = '_nolegend_',color=self.colours[j+1], linewidth=2)
            ax[1,1].plot(Lz_metric, label = f'e = {self.e[j]}, alt = {self.alt[j]} km',color=self.colours[j+1], linewidth=2)
        
        ax[0,0].set_title('R Metric',fontsize=16)
        ax[0,1].set_title('V Metric',fontsize=16)
        ax[1,0].set_title('$r \cdot v$ Metric',fontsize=16)
        ax[1,1].set_title('$L_z$ Metric',fontsize=16)
        ax[1,0].set_xticks([0, self.num_orbits*80, self.num_orbits*160, self.num_orbits*240, self.num_orbits*320, self.num_orbits*400, self.num_orbits*480, self.num_orbits*560, self.num_orbits*640, self.num_orbits*720, self.num_orbits*800],[0, self.num_orbits*0.1, self.num_orbits*0.2, self.num_orbits*0.3, self.num_orbits*0.4, self.num_orbits*0.5, self.num_orbits*0.6, self.num_orbits*0.7, self.num_orbits*0.8, self.num_orbits*0.9, self.num_orbits*1.00])
        ax[1,1].set_xticks([0, self.num_orbits*80, self.num_orbits*160, self.num_orbits*240, self.num_orbits*320, self.num_orbits*400, self.num_orbits*480, self.num_orbits*560, self.num_orbits*640, self.num_orbits*720, self.num_orbits*800],[0, self.num_orbits*0.1, self.num_orbits*0.2, self.num_orbits*0.3, self.num_orbits*0.4, self.num_orbits*0.5, self.num_orbits*0.6, self.num_orbits*0.7, self.num_orbits*0.8, self.num_orbits*0.9, self.num_orbits*1.00])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0,0].legend(by_label.values(), by_label.keys())
        plt.savefig(f"metric-{self.fig_handle}.svg", format="svg", dpi=1200)