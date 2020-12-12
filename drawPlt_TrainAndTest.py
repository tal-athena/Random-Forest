# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

# %matplotlib inline

# Set the style
plt.style.use('fivethirtyeight')

def draw(srtRootDirectory, featuresTest, featuresTrain):
    # Set up the plotting layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
    fig.autofmt_xdate(rotation = 45)

    # Actual max temperature measurement
    ax1.plot(featuresTrain['ED_ENC_NUM'], featuresTrain['_target_'])
    ax1.set_xlabel(''); ax1.set_ylabel('_target_'); ax1.set_title('TargetForTrain')

    # Temperature from 1 day ago
    ax2.plot(featuresTrain['ED_ENC_NUM'], featuresTrain['Proc1SVM'])
    ax2.set_xlabel(''); ax2.set_ylabel('_target_'); ax2.set_title('Proc1SVMForTrain')

    # Temperature from 2 days ago
    ax3.plot(featuresTest['ED_ENC_NUM'], featuresTest['_target_'])
    ax3.set_xlabel('ED_ENC_NUM'); ax3.set_ylabel('_target_'); ax3.set_title('TargetForTest')

    # Friend Estimate
    ax4.plot(featuresTest['ED_ENC_NUM'], featuresTest['Proc1SVM'])
    ax4.set_xlabel('ED_ENC_NUM'); ax4.set_ylabel('_target_'); ax4.set_title('Proc1SVMForTest')

    plt.tight_layout(pad=2)
    plt.savefig(srtRootDirectory + '\\OutputImages\\TestandTradinData.png', bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')