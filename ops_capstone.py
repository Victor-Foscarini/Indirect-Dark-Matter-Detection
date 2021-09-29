import pandas as pd
import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import curve_fit
import pylab
from IPython.display import display, HTML
import unicodedata
import warnings
warnings.filterwarnings("ignore")

class Curves:
    
    random_state=42
    
    def __init__(self,data_folder='data/Begeman et al./',
                 galaxies= ['ddo154','ddo170','ngc1560','ngc2403','ngc2841','ngc2903','ngc3109','ngc3198',
                            'ngc6503','ngc7331','ugc2259']):
        
        #methods
        self.method_name = {'scipy':'least squares'}
        
        #data and galaxies
        self.data_folder = data_folder
        self.galaxies = galaxies
        
        #models and test results
        self.model = []
        self.model_name = []
        #self.result = []
        
    def add_model(self,model_name=None,model_params=[[0,2,2],[1,1,3]]):
        
        #first check known models
        if model_name == 'simple':
            def v(r,R0,P0): return  P0 * np.sqrt( R0**2 * (1-R0/r*np.arctan(r/R0)) )
            self.model.append(v)
            self.model_name.append('simple')
            #self.result.append([])
            print('SUCESS to create {} model'.format('simple'))

        
        elif model_name == 'NFW':
            def v(r,R0,P0) : return P0 * np.sqrt( R0**3/r * ( np.log(1+r/R0) - r/(R0+r) ) )
            self.model.append(v)
            self.model_name.append('NFW')
            #self.result.append([])
            print('SUCESS to create {} model'.format('NFW'))
        
        elif model_name == 'traditional':
            def v(r,R0,P0) : return (R0*P0)/np.sqrt(r)
            self.model.append(v)
            self.model_name.append('traditional')
            print('SUCESS to create {} model'.format('traditional'))
        
        #then try to create models from given params
        else:
            for params in model_params:
                try:
                    r,R0,P0 = sp.Symbol('r'),sp.Symbol('R0'),sp.Symbol('P0')
                    g,a,b = params
                    
                    integral = sp.lambdify([r,R0,P0],
                                           sp.integrate(r**2/(((r/R0)**g)*((1+(r/R0)**a)**((b-g)/a))),r),'numpy')
                    
                    def v(r,R0,P0): return P0 * np.sqrt(1/r*(integral(r,R0,P0)-integral(0,R0,P0)))
                    self.model.append(v)
                    self.model_name.append('{},{},{}'.format(g,a,b))
                    #self.result.append([])
                    print('SUCESS to create model from params: g={},a={},b={}'.format(g,a,b))
                except:
                    print('FAILED to create model from params: g={},a={},b={}'.format(g,a,b))
                    
    def fit_galaxy(self,galaxy='ngc2403',method='scipy',return_plot=True,return_results=False,return_tests=False,
                   return_plot_data=False,guess=None,guess_scale=False,figsize=(16,9)):
        
        #functions for fit and test
        def chi_test(obs,expt,error):
            return np.sum((obs-expt)**2/error**2)
        def chi_test_degrees(obs,expt,error,params=2):
            return np.sum((obs-expt)**2/error**2)/(len(obs)-params-1)
        def residuals(obs,expt):
            return obs-expt
        
        
        #reading file
        df = pd.read_csv(self.data_folder+galaxy+'_with_errors.csv',sep=',')
        df = df.drop(df.columns[[0]],axis=1)
        
        results = [[],[]]
        
        #initial guess proportional to median velocity value
        if not guess:
            guess = [df['vel'][int(len(df)/2)]*(1/100),df['vel'][int(len(df)/2)]*(50)]
        elif not guess_scale:
            guess = [df['vel'][int(len(df)/2)]*guess[0],df['vel'][int(len(df)/2)]*guess[1]]
        elif guess_scale:
            guess = [df['vel'][int(len(df)/2)]*(guess[0]/100),df['vel'][int(len(df)/2)]*(guess[1]*50)]

        #perform curve fitting for each model
        popts = []
        yfits = []
        chi_tests = []
        chi_degrees = []
        residuals_tests = []
        pcovs = []
        if method == 'scipy':
            for i in range(len(self.model)):
                try:
                    popt,pcov = curve_fit(self.model[i],df['rad'],df['vel'],guess,sigma=df['vel_error'],absolute_sigma=True)
                    popts.append(popt)
                    pcovs.append(pcov)
                    yfits.append( self.model[i](df['rad'],*popts[i]) ) #parameter fit
                    chi_tests.append( chi_test(df['vel'],[self.model[i](r,*popts[i]) for r in df['rad']],df['vel_error']) )
                    chi_degrees.append( chi_test_degrees(df['vel'],[self.model[i](r,*popts[i]) for r in df['rad']],df['vel_error']) )
                    residuals_tests.append( residuals(df['vel'],[self.model[i](r,*popts[i]) for r in df['rad']]) )
                except:
                    popt = [None,None]
                    popts.append(popt)
                    yfits.append( [None]*len(df['rad']) )
                    chi_tests.append( None )
                    chi_degrees.append( None )
                    residuals_tests.append( [None]*len(df['rad']) )
        
        elif method == 'fit_method':
            pass
        
        if return_plot_data:
            return df,popts,yfits,chi_tests,chi_degrees,residuals_tests
        
        if return_plot:
            
            ##plot fits
            
            #palette = ['darkblue','darkgreen','darkred','darkcyan','darkmagenta','yellowish','cool grey']
            palette = ['blue','green','red','black','magenta','yellow','grey']
            
            #plot data
            
            plt.figure(figsize=(16,9))
            #plt.title(galaxy.upper()+": ajustes [{} - {}]".format(self.data_folder[5:-1],self.method_name[method]),fontsize=1.5*figsize[0])
            
            plt.xlabel('Radius (kpc)',fontsize=1.875*figsize[0])
            plt.ylabel('Velocity (km s⁻¹)',fontsize=1.875*figsize[0])
            
            plt.xticks(fontsize=1.25*figsize[0])
            plt.yticks(fontsize=1.25*figsize[0])

            #data used with errors
            plt.errorbar(df['rad'],df['vel'],yerr=df['vel_error'],fmt='o',color='black',alpha=0.2)
            
            #plot models
            for i in range(len(self.model)):
                plt.plot(df['rad'],yfits[i],linewidth=5,label=self.model_name[i],color=palette[i],alpha=0.5)

            plt.xlim(0,df['rad'].iloc[-1]*1.25)
            plt.legend(fontsize=1.5*figsize[0],loc='upper right') #loc='lower right' para NGC2903 e NGC6503
            plt.show()
            #ba
            #plot residuals
            
            plt.figure(figsize=figsize)
            #plt.title(galaxy.upper()+": resíduos e testes [{} - {}]".format(self.data_folder[5:-1],self.method_name[method]),fontsize=1.5*figsize[0])

            plt.xlabel("Radius (kpc)",fontsize=1.875*figsize[0])
            plt.ylabel("Residual (km s⁻¹)",fontsize=1.875*figsize[0])

            plt.xticks(fontsize=1.25*figsize[0])
            plt.yticks(fontsize=1.25*figsize[0])
            
            try:
                max_y = max([abs(x) if x else 0 for y in residuals_tests for x in y ])*1.5
                plt.ylim(-max_y,max_y)
            except:
                pass
            
            plt.xlim(0,df['rad'].iloc[-1]*1.3)
                        
            plt.plot(np.arange(-1,int(df['rad'].iloc[-1]*2)),[0]*len(np.arange(-1,int(df['rad'].iloc[-1]*2))),
                     color='black')
            for i in range(len(self.model)):
                plt.scatter(df['rad'],residuals_tests[i],linewidth=5,label=self.model_name[i],color=palette[i],alpha=0.4)
            
            #try to plot results togheter
            y_pos = np.linspace(0,-max_y,5*len(self.model)+1)
            for i in range(len(self.model)):
                if self.model_name[i] == 'traditional':
                    try:
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+3],
                                 unicodedata.lookup('greek small letter chi')+'²: {}'.format(round(chi_tests[i],2)),
                                 fontsize=4*figsize[1]/len(self.model),color=palette[i])
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+4],
                                 unicodedata.lookup('greek small letter chi')+'²/n: {}'.format(round(chi_degrees[i],2)),
                                 fontsize=4*figsize[1]/len(self.model),color=palette[i])
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+2],
                                 'constante: {}'.format(round(popts[i][0]*popts[i][1],2)),
                                 fontsize=4*figsize[1]/len(self.model),color=palette[i])
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+1],'Model {}'.format(self.model_name[i]),
                                 fontsize=6*figsize[1]/len(self.model),color=palette[i])
                    except:
                        pass
                else:
                    try:
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+3],
                                 unicodedata.lookup('greek small letter chi')+'²: {}'.format(round(chi_tests[i],2)),
                                 fontsize=4*figsize[1]/len(self.model),color=palette[i])
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+4],
                                 unicodedata.lookup('greek small letter chi')+'²/n: {}'.format(round(chi_degrees[i],2)),
                                 fontsize=4*figsize[1]/len(self.model),color=palette[i])
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+2],
                                 '(R0, P0): ({}, {})'.format(round(popts[i][0],2),round(popts[i][1],2)),
                                 fontsize=4*figsize[1]/len(self.model),color=palette[i])
                        plt.text(df['rad'].iloc[-1]*1.05,y_pos[5*i+1],'Model {}'.format(self.model_name[i]),
                                 fontsize=6*figsize[1]/len(self.model),color=palette[i])     
                    except:
                        pass

            
            plt.legend(fontsize=1.5*figsize[0],loc='upper right') #loc='lower right' para NGC2903 e NGC6503
            plt.show()
            
#             #plot test results
            
#             plt.figure(figsize=(16,9))
#             plt.title('Resultados dos ajustes',fontsize=2.5*figsize[0])
#             y_pos = np.linspace(0.9,0.1,4*len(self.model))
#             #print('RESULTADOS:\n')
#             for i in range(len(self.model)):
#                 #print('Model: {}'.format(self.model_name[i]))
#                 #print('Qui sem graus: {}'.format(chi_tests[i]))
#                 #print('Qui com graus: {}'.format(chi_degrees[i]))
#                 #print('\n')
#                 plt.text(0.1,y_pos[4*i],'Model {}'.format(self.model_name[i]),fontsize=5*figsize[1]/len(self.model),color=palette[i])
#                 plt.text(0.1,y_pos[4*i+1],'Qui quadrado sem graus: {}'.format(chi_tests[i]),fontsize=5*figsize[1]/len(self.model),color=palette[i])
#                 plt.text(0.1,y_pos[4*i+2],'Qui quadrado com graus: {}'.format(chi_degrees[i]),fontsize=5*figsize[1]/len(self.model),color=palette[i])
            
#             plt.axis('off')
#             plt.show()
        
        print('chi_test: ',chi_degrees)
        param_errors = []
        for pcov in pcovs:
            try:
                param_errors.append(np.sqrt(np.diag(pcov)))
            except:
                param_errors.append([None, None])
        
        print('params: ')
        for i in [dict(zip(popt, param_error))for popt,param_error in zip(popts, param_errors)]:
            print('{} +- {}'.format(round(list(i.keys())[0],2), round(list(i.values())[0],2)))
            print('{} +- {}'.format(round(list(i.keys())[1],2), round(list(i.values())[1],2)))
            print()
        
        #print('popts: ',popts)
        #print('param_errors: ',param_errors)


        if return_results:
            return [ [chi_test,chi_degree,popt] for (chi_test,chi_degree,popt) in zip(chi_tests,chi_degrees,popts) ]
        
        if return_tests:
            return [ chi_degree for chi_degree in chi_degrees ]
    
    def fit_all_galaxies(self,method='scipy',galaxies=None,
                         return_plot=True,return_results=False,return_tests=False,
                         x_len=None,y_len=None,figsize=None):
        
        if not galaxies:
            galaxies = self.galaxies
        
        if not x_len or not y_len:
            y_len,x_len = len(galaxies),2
            
        if not figsize:
            figsize=(16*x_len,9*y_len)
        
        fig,axes = plt.subplots(y_len,x_len,figsize=figsize)
        
        for j in range(len(galaxies)):
            
            y_ax = j
            x_ax = 0
            
            df,popts,yfits,chi_tests,chi_degrees,residuals_tests = self.fit_galaxy(method='scipy',
                                                                                  galaxy=galaxies[j],
                                                                                  return_plot=False,
                                                                                  return_plot_data=True,
                                                                                  figsize=(16,9))
            
            palette = ['blue','green','red','cyan','magenta','yellow','grey']
            
            #plot data
            
            axes[y_ax,x_ax].set_title(galaxies[j].upper()+": ajustes [{} - {}]".format(
                self.data_folder[5:-1],self.method_name[method]),fontsize=1.5*figsize[0]/x_len)
            
            axes[y_ax,x_ax].set_xlabel("Raio (kpc)",fontsize=1.875*figsize[0]/x_len)
            axes[y_ax,x_ax].set_ylabel('Velocidade (km s⁻¹)',fontsize=1.875*figsize[0]/x_len)
            
            #axes[x_ax,y_ax].set_xticks()
            #axes[x_ax,y_ax].set_yticks()
            
            #data used with errors
            axes[y_ax,x_ax].errorbar(df['rad'],df['vel'],yerr=df['vel_error'],fmt='o',color='black',alpha=0.2)
            
            #plot models
            for i in range(len(self.model)):
                axes[y_ax,x_ax].plot(df['rad'],yfits[i],linewidth=5,label=self.model_name[i],color=palette[i],alpha=0.4)
                
            axes[y_ax,x_ax].set_xlim(0,df['rad'].iloc[-1]*1.25)
            axes[y_ax,x_ax].legend(fontsize=1.5*figsize[0]/x_len,loc='lower right') #loc='lower right' para NGC2903 e NGC6503
        
            x_ax+=1

            #plot residuals
            axes[y_ax,x_ax].set_title(galaxies[j].upper()+": residuals and tests [{} - {}]".format(self.data_folder[5:-1],self.method_name[method]),
                                  fontsize=1.5*figsize[0]/x_len)

            axes[y_ax,x_ax].set_xlabel("Radius (kpc)",fontsize=1.875*figsize[0]/x_len)
            axes[y_ax,x_ax].set_ylabel("Residual (km s⁻¹)",fontsize=1.875*figsize[0]/x_len)

            #axes[y_ax,x_ax].xticks(fontsize=1.25*figsize[0])
            #axes[y_ax,x_ax].yticks(fontsize=1.25*figsize[0])

            try:
                max_y = max([abs(x) if x else 0 for y in residuals_tests for x in y ])*1.5
                axes[y_ax,x_ax].set_ylim(-max_y,max_y)
            except:
                pass

            axes[y_ax,x_ax].set_xlim(0,df['rad'].iloc[-1]*1.3)

            axes[y_ax,x_ax].plot(np.arange(-1,int(df['rad'].iloc[-1]*2)),[0]*len(np.arange(-1,int(df['rad'].iloc[-1]*2))),
                     color='black')
            for i in range(len(self.model)):
                axes[y_ax,x_ax].scatter(df['rad'],residuals_tests[i],linewidth=5,label=self.model_name[i],color=palette[i],alpha=0.4)

            #try to plot results togheter
            y_pos = np.linspace(0,-max_y,5*len(self.model)+1)
            for i in range(len(self.model)):
                try:
                    axes[y_ax,x_ax].text(df['rad'].iloc[-1]*1.05,y_pos[5*i+3],
                                         unicodedata.lookup('greek small letter chi')+'²: {}'.format(round(chi_tests[i],2)),
                             fontsize=4*figsize[1]/len(self.model)/y_len,color=palette[i])
                    axes[y_ax,x_ax].text(df['rad'].iloc[-1]*1.05,y_pos[5*i+4],
                                         unicodedata.lookup('greek small letter chi')+'²/n: {}'.format(round(chi_degrees[i],2)),
                             fontsize=4*figsize[1]/len(self.model)/y_len,color=palette[i])
                    axes[y_ax,x_ax].text(df['rad'].iloc[-1]*1.05,y_pos[5*i+2],
                             '(R0, P0): ({}, {})'.format(round(popts[i][0],2),round(popts[i][1],2)),
                             fontsize=4*figsize[1]/len(self.model)/y_len,color=palette[i])
                    axes[y_ax,x_ax].text(df['rad'].iloc[-1]*1.05,y_pos[5*i+1],'Model {}'.format(self.model_name[i]),
                             fontsize=6*figsize[1]/len(self.model)/y_len,color=palette[i])     
                except:
                    pass

                axes[y_ax,x_ax].legend(fontsize=1.5*figsize[0]/x_len,loc='upper right') #loc='lower right' para NGC2903 e NGC6503
         
        #plt.savefig('all_plots.png')
        plt.show()
        
        
    def test_initialparams(self,method='scipy',galaxy='ngc2403',error='chi_square_with_degree',
                           vary=[0.001,0.01,0.1,1,10,100,1000],figsize=(12,12),guess_scale=False):
        
        tests = [pd.DataFrame(np.zeros((len(vary),len(vary)),dtype=float),columns=vary,index=vary) for _ in range(len(self.model))]

        if error == 'chi_square_with_degree':
            for i in vary:
                for j in vary:
                    test_ij = self.fit_galaxy(galaxy=galaxy,return_plot=False,return_tests=True,
                                              guess=(i,j),guess_scale=guess_scale)
                    
                    for k in range(len(tests)):
                        tests[k][i][j] = test_ij[k][1]
                    
        #plot results 
        for i in range(len(self.model)):
            plt.figure(figsize=figsize)
            plt.title(galaxy.upper()+': estimates {} [{} - {}}]'.format(self.model_name[i],self.data_folder[5:-1],self.method_name[method]),fontsize=1*figsize[0])
            sns.heatmap(tests[i],cmap='coolwarm',
                        annot=True,annot_kws={"fontsize":1*figsize[0]})
            
            plt.xticks(fontsize=1.25*figsize[0])
            plt.yticks(fontsize=1.25*figsize[0])
            
            plt.xlabel("parameter: R0 [in scale 0.01 kpc]",fontsize=1.875*figsize[0])
            plt.ylabel("parameter: P0 [in scale 50 km s⁻¹ kpc⁻¹]",fontsize=1.875*figsize[0])
            
            plt.show()
            
    def test_all_initialparams(self,method='scipy',error='chi_square_with_degree',
                               galaxies=None,vary=[0.001,0.01,0.1,1,10,100,1000],
                               figsize=None,guess_scale=False):
        
        if not galaxies:
            galaxies = self.galaxies
            
        y_len,x_len = len(galaxies),len(self.model)
            
        if not figsize:
            figsize=(16*x_len,7*y_len)
        
        fig,axes = plt.subplots(y_len,x_len,figsize=figsize)
        palette = ['blue','green','red','cyan','magenta','yellow','grey']
        
        for y_ax in range(len(galaxies)):
            
            tests = [pd.DataFrame(np.zeros((len(vary),len(vary)),dtype=float),columns=vary,index=vary) for _ in range(len(self.model))]

            if error == 'chi_square_with_degree':
                for i in vary:
                    for j in vary:
                        test_ij = self.fit_galaxy(galaxy=galaxies[y_ax],return_plot=False,return_tests=True,
                                                  guess=(i,j),guess_scale=guess_scale)

                        for k in range(len(tests)):
                            tests[k][i][j] = test_ij[k]

            #plot results
            for x_ax in range(len(self.model)):
                
                sns.heatmap(tests[x_ax],cmap='coolwarm',
                            annot=True,annot_kws={'fontsize':8.75/len(vary)*figsize[0]/x_len},
                            ax=axes[y_ax,x_ax])
                
                axes[y_ax,x_ax].set_title(galaxies[y_ax].upper()+': estimativas {} [{} - {}]'.format(
                    self.model_name[x_ax],self.data_folder[5:-1],self.method_name[method]),fontsize=1*figsize[0]/x_len)
                
                #plt.xticks(fontsize=1.25*figsize[0])
                #plt.yticks(fontsize=1.25*figsize[0])
                
                axes[y_ax,x_ax].set_xlabel("parameter: R0 [in scale 0.01 kpc]",fontsize=1.5*figsize[0]/(x_len*2))
                axes[y_ax,x_ax].set_ylabel("parameter: P0 [in scale 50 km s⁻¹ kpc⁻¹]",fontsize=1.5*figsize[0]/(x_len*2))
                
        
        #plt.savefig('all_initalparams.png')
        plt.show()
    