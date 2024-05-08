import matplotlib.pyplot as plt

def plot_heatmap_and_errors(Q_LO_ind, Q_HO_ind, Q_HO_sol, L2_Error):
    fig, ax = plt.subplots(3, 3)

    im0 = ax[0, 0].matshow(Q_LO_ind[23, 0, :, :, 0], cmap='jet')
    im1 = ax[0, 1].matshow(Q_HO_ind[23, 0, :, :, 0], cmap='jet')
    im2 = ax[0, 2].matshow(Q_HO_sol[48, 23, 0, :, :, 0], cmap='jet')

    im3 = ax[1, 0].matshow(Q_LO_ind[23, 0, :, :, 1], cmap='jet')
    im4 = ax[1, 1].matshow(Q_HO_ind[23, 0, :, :, 1], cmap='jet')
    im5 = ax[1, 2].matshow(Q_HO_sol[48, 23, 0, :, :, 1], cmap='jet')

    im6 = ax[2, 0].matshow(Q_LO_ind[23, 0, :, :, 2], cmap='jet')
    im7 = ax[2, 1].matshow(Q_HO_ind[23, 0, :, :, 2], cmap='jet')
    im8 = ax[2, 2].matshow(Q_HO_sol[48, 23, 0, :, :, 2], cmap='jet')

    cbar2 = fig.colorbar(im2, ax=ax[0, :])
    cbar5 = fig.colorbar(im5, ax=ax[1, :])
    cbar8 = fig.colorbar(im8, ax=ax[2, :])

    fig.text(0.1, 0.8, 'Rhou', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.5, 'Rhov', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.2, 'Rhow', va='center', ha='right', rotation='vertical')

    ax[0, 0].set_title('Original LO')
    ax[0, 1].set_title('Original HO')
    ax[0, 2].set_title('Predicted HO')

    for axes_row in ax:
        for axes in axes_row:
            axes.set_xticks([])
            axes.set_yticks([])

    plt.savefig("NEURALNET/HeatMap.pdf")

    constant_value = 0.1765
    Interp_Error = np.full(len(L2_Error), constant_value)

    fig = plt.figure()
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(0.002 * np.linspace(0, len(L2_Error), len(L2_Error)), L2_Error, label='SRCNN')
    plt.semilogy(0.002 * np.linspace(0, len(L2_Error), len(L2_Error)), Interp_Error, label='Cubic Interpolation')
    plt.legend()
    plt.show()

def plot_reconstructed(Q_HO_ind, Q_HO_SRCNN, Q_HO_SRGAN):
    fig, ax = plt.subplots(3, 3, figsize=(18, 18))  # Set desired figure size and subplot size

    # Define the extent for imshow based on the array dimensions
    extent = [0, Q_HO_SRCNN.shape[1], 0, Q_HO_SRCNN.shape[2]]  # Assuming square shape

    im1 = ax[0, 0].imshow(Q_HO_ind[10, :, :, 0], cmap='jet', extent=extent)
    im2 = ax[0, 1].imshow(Q_HO_SRCNN[10, :, :, 0], cmap='jet', extent=extent)
    im3 = ax[0, 2].imshow(Q_HO_SRGAN[10, :, :, 0], cmap='jet', extent=extent)

    im4 = ax[1, 0].imshow(Q_HO_ind[10, :, :, 1], cmap='jet', extent=extent)
    im5 = ax[1, 1].imshow(Q_HO_SRCNN[10, :, :, 1], cmap='jet', extent=extent)
    im6 = ax[1, 2].imshow(Q_HO_SRGAN[10, :, :, 1], cmap='jet', extent=extent)

    im7 = ax[2, 0].imshow(Q_HO_ind[10, :, :, 2], cmap='jet', extent=extent)
    im8 = ax[2, 1].imshow(Q_HO_SRCNN[10, :, :, 2], cmap='jet', extent=extent)
    im9 = ax[2, 2].imshow(Q_HO_SRGAN[10, :, :, 2], cmap='jet', extent=extent)

    # Add colorbars
    cbar2 = fig.colorbar(im2, ax=ax[0, :])
    cbar5 = fig.colorbar(im5, ax=ax[1, :])
    cbar8 = fig.colorbar(im8, ax=ax[2, :])

    # Set titles and axis labels
    fig.text(0.1, 0.8, 'Rhou', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.5, 'Rhov', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.2, 'Rhow', va='center', ha='right', rotation='vertical')

    ax[0, 0].set_title('Original')
    ax[0, 1].set_title('SRCNN')
    ax[0, 2].set_title('SRGAN')

    # Remove ticks
    for axes_row in ax:
        for axes in axes_row:
            axes.set_xticks([])
            axes.set_yticks([])

    # Save and display plot
    plt.savefig("NEURALNET/HeatMap_Reconstructed.pdf")
    plt.show()