import numpy as np
import os
import matplotlib.pyplot as plt
import healpy as hp
import tensorflow as tf
import foscat.alm as foscat_alm
import foscat.Synthesis as synthe

class cmbscat:
    """
    Performs scattering covariance synthesis on a dataset of Q,U polarization input maps.
    Note that by default it assumes the single-target approach (see Campeti et al. 2025). 
    This means the synthesis should be repeated running this script for every target in the input dataset. Then  
    then targets should be uniformly sampled with replacement from the input dataset, and, for each sampled target, a synthesized map should be selected from the corresponding batch. If the same target is sampled more than once, you should ensure that each time a different syntehsized map is selected from the same batch.
    """

    def __init__(self, params):
        """
        Initialize cmbscat with a dictionary of parameters.
        
        Args:
            params (dict): Dictionary containing the parameters needed for the script.
        """
        # Store parameters
        self.params = params
        
        # Extract parameters (with defaults if needed)
        self.NNN            = params.get('NNN') # number of maps of the reference dataset
        self.gauss_real     = params.get('gauss_real') # if True generates gaussian realization from a reference covariance as input dataset, else uses directly the input maps. 
        self.NGEN           = params.get('NGEN') # number of maps in a batch for mean-field gradient descent
        self.n_new_samples  = params.get('n_new_samples') # number of samples in the input dataset
        self.index_ref      = params.get('index_ref') # indices of input maps considere
        self.seed           = params.get('seed') # list of seeds for the intialization of the batch for gradient descent
        self.nmask          = params.get('nmask') # number of masks used
        self.mask           = params.get('mask', None) # mask
        self.nside          = params.get('nside') # nside desired
        self.NORIENT        = params.get('NORIENT', 4) # number of orientations used in the SC 
        self.new_start      = params.get('new_start', 0) # first target in input dataset
        self.new_finish     = params.get('new_finish', 1) # last target in input dataset
        self.cov            = params.get('cov', True) # whether to use SC or ST
        self.no_orient      = params.get('no_orient', False) # if True doesn't use the orientation matrices
        self.nstep          = params.get('nstep', 1000) # number of steps in gradient descent
        self.KERNELSZ       = params.get('KERNELSZ', 3) # wavelet kernel size in pixels
        self.ave_target     = params.get('ave_target', False) # Wheter to use the average-target strategy. Default is single-target 
        self.outname        = params.get('outname', 'output') # output name for the synthesized maps
        self.outpath        = params.get('outpath', './data/') # output path
        self.data_path      = params.get('data') # path fo input data

        # Depending if you want the scattering transform or the scattering covariance
        # import the appropriate foscat scat module
        if self.cov:
            import foscat.scat_cov as sc
        else:
            import foscat.scat as sc
        self.sc = sc

        # Prepare the alm object for the ps loss
        self.alm = foscat_alm.alm(nside=self.nside)
        
        # Initialize placeholders for data/matrices/scat coefficients
        self.im    = None   # input data after regrading and normalizing
        self.dim   = None   # standard deviation of input maps
        self.mdim  = None   # Mean of input maps
        self.scat_op = None
        
        # Orientation matrices
        self.cmat1  = None
        self.cmat12 = None
        self.cmat2  = None
        self.cmat22 = None
        self.cmatx  = None
        self.cmatx2 = None
        
        # Reference SC dictionaries
        self.ref1 = {}
        self.ref2 = {}
        self.refx = {}
        
        # Storage for power spectra
        self.c_l1 = None
        self.c_l2 = None
        
        print(f"[INIT] CMBSCAT with nside={self.nside}, scat cov={self.cov}, no_orient={self.no_orient}")


    def dodown(self, a, nside):
        """
        Function to reduce data resolution (adapted to nested ordering).
        Args:
            a (np.array): array of size 12 * n_in^2
            nside (int): target nside
        Returns:
            np.array: re-gridded array
        """
        nin = int(np.sqrt(a.shape[0] // 12))
        if nin == nside:
            return a
        return np.mean(a.reshape(12*nside*nside, (nin//nside)**2), axis=1)


    @tf.function
    def dospec(self, im):
        """
        A tf.function to compute the power spectra using foscat_alm.alm.anafast.
        Returns both the L_1 and the L_2 norm angular power spectra.
        Args:
            im (tf.Tensor or np.array): input map shape (n_samples, 2, 12*nside^2) or (2, 12*nside^2)
        Returns:
            (tf.Tensor, tf.Tensor): c_l2, c_l1
        """
        return self.alm.anafast(im, nest=True)


    # -------------------------------------------------------------------------
    # 1) Preprocessing input dataset (downgrade, reorder, normalize)
    # -------------------------------------------------------------------------
    def preprocess_data(self):
        """
        Loads the data from self.data_path, possibly downgrades it to the 
        desired nside, reorders to nest, and stores the result in self.im.
        """
        print(f"[PREPROCESS] Loading data from: {self.data_path}")
        data_in = np.load(self.data_path)
        
        # The script logic: use only Q, U => data_in[:self.NNN, 1:, :]
        # assumes that input data is T,Q,U maps
        im = data_in[:self.NNN, 1:, :] 
        del data_in
        
        nside2 = int(np.sqrt(im.shape[2] // 12))
        idx_nest = hp.nest2ring(self.nside, np.arange(12*self.nside*self.nside))
        
        # Downgrade if needed and reorder from RING to NEST the input data 
        if nside2 != self.nside:
            im2 = np.zeros([self.NNN, 2, 12*self.nside*self.nside])
            for k in range(self.NNN):
                tmp = np.zeros([2, 12*self.nside*self.nside])
                for l in range(2):
                    tmp[l] = hp.ud_grade(im[k, l], self.nside)
                for l in range(2):
                    im2[k, l] = tmp[l, idx_nest]
            im = im2
        else:
            # If same nside, just reorder ring->nest if needed
            im2 = np.zeros([self.NNN, 2, 12*self.nside*self.nside])
            for k in range(self.NNN):
                for l in range(2):
                    im2[k, l, :] = im[k, l, idx_nest]
            im = im2

        print(f"[PREPROCESS] Data shape after regrade/reorder: {im.shape}")
        self.im = im


    # -------------------------------------------------------------------------
    # 2) Generate Gaussian maps from PCA (SVD)
    # -------------------------------------------------------------------------
    def generate_gaussian_maps(self):
        """
        Uses PCA (via SVD) on the loaded dataset to generate random Gaussian maps.
        Overwrites self.im with the new generated maps (self.n_new_samples of them).
        """
        if self.im is None:
            raise ValueError("No data loaded. Please call preprocess_data() first.")

        data = self.im
        n_samples, n_channels, N_pix = data.shape

        # Reshape => (n_samples, n_channels*N_pix)
        data_reshaped = data.reshape(n_samples, -1)

        # Compute mean & center
        m = np.mean(data_reshaped, axis=0)
        data_centered = data_reshaped - m

        # SVD
        print("[GAUSS] Performing SVD for PCA-based map generation...")
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        eigenvalues = (S**2) / (n_samples - 1)
        V = Vt.T

        # Generate random coefficients
        #np.random.seed(42)
        coefficients = np.random.randn(self.n_new_samples, len(eigenvalues))
        scaled_coefficients = coefficients * np.sqrt(eigenvalues)

        # Generate new maps
        new_maps_centered = scaled_coefficients @ V.T
        new_maps_reshaped = new_maps_centered + m
        new_maps = new_maps_reshaped.reshape(self.n_new_samples, n_channels, N_pix)

        # Overwrite self.im with new maps
        self.im = new_maps
        print(f"[GAUSS] Generated {self.n_new_samples} random Gaussian maps of shape {self.im.shape}")


    # -------------------------------------------------------------------------
    # 3) Normalize data (pixel-wise)
    # -------------------------------------------------------------------------
    def normalize_data(self):
        """
        Compute mean and std across the dataset self.im, then normalize.
        Store the mean and std in self.mdim and self.dim for later usage.
        """
        if self.im is None:
            raise ValueError("No data to normalize. Please load/generate data first.")
        
        # shape => (n_samples, n_channels, N_pix)
        self.dim = np.std(self.im, axis=0)   # std map of input dataset shape (n_channels, N_pix)
        self.mdim = np.mean(self.im, axis=0) # mean map of input dataset shape (n_channels, N_pix)
        
        # Normalize
        self.im = (self.im - self.mdim[None, ...]) / self.dim[None, ...]
        print("[NORMALIZE] Data has been normalized (channel-wise mean/std).")


    # -------------------------------------------------------------------------
    # 4) Initialization of orientation matrices
    # -------------------------------------------------------------------------
    def init_scat_and_orientation_matrices(self):
        """
        Initialize the scattering operator self.scat_op and orientation matrices
        (cmat1, cmat12, cmat2, cmat22, cmatx, cmatx2). If no_orient=True, 
        set them to None.
        """
        # Build scattering operator
        print("[INIT_ORIENT] Initializing scattering operator.")
        self.scat_op = self.sc.funct(
            NORIENT=self.NORIENT,
            KERNELSZ=self.KERNELSZ,
            JmaxDelta=0,
            all_type='float64'
        )
        
        if self.no_orient:
            self.cmat1 = self.cmat12 = None
            self.cmat2 = self.cmat22 = None
            self.cmatx = self.cmatx2 = None
            print("[INIT_ORIENT] Orientation disabled. cmat = None.")
            return
        
        # Compute orientation matrices
        upscale_flag = (self.KERNELSZ == 5) # if KERNELSZ=5 upscaling is performed inside HealpixML
        
        print("[INIT_ORIENT] Computing orientation matrices cmat1, cmat12, etc.")
        
        self.cmat1,  self.cmat12  = self.scat_op.stat_cfft(self.im[:, 0, :], 
                                                           upscale=upscale_flag, 
                                                           smooth_scale=0)
        
        self.cmat2,  self.cmat22  = self.scat_op.stat_cfft(self.im[:, 1, :], 
                                                           upscale=upscale_flag, 
                                                           smooth_scale=0)
        
        self.cmatx, self.cmatx2 = self.scat_op.stat_cfft(self.im[:, 0, :], 
                                                          image2=self.im[:, 1, :],
                                                          upscale=upscale_flag, 
                                                          smooth_scale=0)


    # -------------------------------------------------------------------------
    # 5) Compute reference scattering coefficients
    # -------------------------------------------------------------------------
    def init_reference_scat(self):
        """
        For each map in self.im, compute scattering coefficients (Q, U, cross).
        Also compute the average and std of the power spectra (c_l1, c_l2).
        Store them as class attributes for later usage in losses.
        """
        im = self.im
        scat_op = self.scat_op
        
        n_maps = im.shape[0]
        self.ref1 = {}
        self.ref2 = {}
        self.refx = {}
        
        print("[INIT_REF_SCAT] Computing reference scattering for each map.")
        for k in range(n_maps):
            # Q channel
            self.ref1[k] = scat_op.eval(im[k, 0], norm='self',
                                        cmat=self.cmat1, cmat2=self.cmat12)
            
            # U channel
            self.ref2[k] = scat_op.eval(im[k, 1], norm='self',
                                        cmat=self.cmat2, cmat2=self.cmat22)
            
            # Cross (Q,U)
            self.refx[k] = scat_op.eval(im[k, 0], image2=im[k, 1], norm='self',
                                        cmat=self.cmatx, cmat2=self.cmatx2)

        print("[INIT_REF_SCAT] Done computing reference scattering.")


    # -------------------------------------------------------------------------
    # 5) Compute reference angular power spectra
    # -------------------------------------------------------------------------
    def init_reference_ps(self):
        """
        For each map in self.im, compute angular power spectra using the built in anafast in tensorflow. 
        Also compute the average and std of the power spectra (c_l1, c_l2).
        Store them as class attributes for later usage in losses.
        The input map should be un-normalized at this step.
        """
        im = self.im

        n_maps = im.shape[0]
        
        self.c_l1 = np.zeros([n_maps, 3, 3*self.nside])
        self.c_l2 = np.zeros([n_maps, 3, 3*self.nside])

        print("[INIT_REF_SCAT] Computing angular power spectra for each map.")
        for k in range(n_maps):
            # Power spectra of from un-normalized input maps if needed
            tp_l2, tp_l1 = self.dospec(im[k])
            self.c_l1[k] = tp_l1.numpy()
            self.c_l2[k] = tp_l2.numpy()

        print("[INIT_REF_SCAT] Done computing reference power spectra.")



    # -------------------------------------------------------------------------
    # 6) Define loss functions
    # -------------------------------------------------------------------------
    def The_loss_spec(self, x, scat_operator, args):
        """
        Loss function that compares the power spectrum of current synthesis vs. reference.

        Args:
            x   (tf.Tensor): shape (batch, 2, 12*nside^2)
            scat_operator  : used for backend
            args           : (mean_val, std_val, r_c_l1, r_c_l2, d_c_l1, d_c_l2, alm)
        Returns:
            loss (tf.Tensor)
        """
        mean_val = args[0]
        std_val  = args[1]
        r_c_l1   = args[2]
        r_c_l2   = args[3]
        d_c_l1   = args[4]
        d_c_l2   = args[5]

        tp_c_l2, tp_c_l1 = self.dospec(x[0]*std_val + mean_val)
        c_l1 = tp_c_l1 - r_c_l1
        c_l2 = tp_c_l2 - r_c_l2

        for k in range(1, x.shape[0]):
            tp_c_l2, tp_c_l1 = self.dospec(x[k]*std_val + mean_val)
            c_l1 = c_l1 + tp_c_l1 - r_c_l1
            c_l2 = c_l2 + tp_c_l2 - r_c_l2
        
        bk = scat_operator.backend
        loss = bk.bk_reduce_mean(bk.bk_square(c_l1/d_c_l1)) + \
               bk.bk_reduce_mean(bk.bk_square(c_l2/d_c_l2))
        return loss


    def The_loss(self, x, scat_operator, args):
        """
        Auto-SC scattering loss.
        Args:
            x   : (batch, 2, 12*nside^2)
            scat_operator
            args: (ref, sref, cmat, cmat2, pol_index)
        """
        ref   = args[0]
        sref  = args[1]
        cmat  = args[2]
        cmat2 = args[3]
        p     = args[4]

        learn = scat_operator.eval(x[:, p], norm='self', cmat=cmat, cmat2=cmat2)
        learn = scat_operator.reduce_sum_batch(learn)
        loss = scat_operator.reduce_mean(
            scat_operator.square((learn - x.shape[0] * ref) / sref)
        )
        return loss


    def The_loss_x(self, x, scat_operator, args):
        """
        Cross-SC scattering loss.
        Args:
            x   : (batch, 2, 12*nside^2)
            scat_operator
            args: (refx, srefx, cmatx, cmatx2)
        """
        refx   = args[0]
        srefx  = args[1]
        cmatx  = args[2]
        cmatx2 = args[3]

        learn = scat_operator.eval(x[:, 0], image2=x[:, 1], norm='self',
                                   cmat=cmatx, cmat2=cmatx2)
        learn = scat_operator.reduce_sum_batch(learn)
        loss = scat_operator.reduce_mean(
            scat_operator.square((learn - x.shape[0] * refx) / srefx)
        )
        return loss


    # -------------------------------------------------------------------------
    # 7) Looping over index_ref to run the synthesis for each target map
    # -------------------------------------------------------------------------
    def loop_synthesis(self):
        """
        Main loop that builds losses for each reference map (iref), 
        runs the Synthesis, and saves results.
        """
        # Precompute average and std of power spectra c_l1, c_l2
        r_c_l1 = np.mean(self.c_l1, axis=0)
        r_c_l2 = np.mean(self.c_l2, axis=0)
        
        d_c_l1 = np.std(self.c_l1, axis=0)
        d_c_l2 = np.std(self.c_l2, axis=0)
        
        # Original script sets first two multipoles to 1
        d_c_l1[:, 0:2] = 1.0
        d_c_l2[:, 0:2] = 1.0

        # Moments for ref1, ref2, refx
        mref1, vref1 = self.scat_op.moments(self.ref1)
        mref2, vref2 = self.scat_op.moments(self.ref2)
        mrefx, vrefx = self.scat_op.moments(self.refx)
        
        # if mask is None fill it with 1s
        mask = (np.ones([1,12*self.nside**2]) 
                if self.mask is None else self.mask)


        # loop over the target input maps 
        for iref in self.index_ref[self.new_start : self.new_finish]:
            first = True
            f_outname = f'{self.outname}_{iref:03d}'
            
            # Storage for final maps & losses
            allmap = np.zeros([len(self.seed[iref]), self.im.shape[1], self.im.shape[2]])
            floss  = np.zeros([len(self.seed[iref])])

            if self.ave_target:
                # use average SC coeff of the input dataset as target
                tmp1 = mref1
                tmp2 = mref2
                tmpx = mrefx
            else:
                # use current (iref) input map SC coeff as target 
                tmp1 = self.ref1[iref]
                tmp2 = self.ref2[iref]
                tmpx = self.refx[iref]

            # Build single-pol losses
            loss1 = synthe.Loss(
                self.The_loss, self.scat_op,
                tmp1, vref1, 
                self.cmat1, self.cmat12, 0
            )
            loss2 = synthe.Loss(
                self.The_loss, self.scat_op,
                tmp2, vref2, 
                self.cmat2, self.cmat22, 1
            )
            # Cross-pol loss
            lossx = synthe.Loss(
                self.The_loss_x, self.scat_op,
                tmpx, vrefx,
                self.cmatx, self.cmatx2
            )

            if self.ave_target:
                # Power spectrum loss with average-target
                loss_sp = synthe.Loss(
                    self.The_loss_spec, self.scat_op,
                    self.mdim, self.dim,
                    self.scat_op.backend.bk_cast(r_c_l1),
                    self.scat_op.backend.bk_cast(r_c_l2),
                    self.scat_op.backend.bk_cast(d_c_l1),
                    self.scat_op.backend.bk_cast(d_c_l2),
                    self.alm  # or self.alm if you want
                    )

            else:    
                # Power spectrum loss with single-target
                loss_sp = synthe.Loss(
                    self.The_loss_spec, self.scat_op,
                    self.mdim, self.dim,
                    self.scat_op.backend.bk_cast(self.c_l1[iref]),
                    self.scat_op.backend.bk_cast(self.c_l2[iref]),
                    self.scat_op.backend.bk_cast(d_c_l1),
                    self.scat_op.backend.bk_cast(d_c_l2),
                    self.alm  # or self.alm if you want
                    )

            # Combine all losses
            sy = synthe.Synthesis([loss1, loss2, lossx, loss_sp])

            # Loop over random seeds
            for iseed in range(0, len(self.seed[iref]), self.NGEN):
                np.random.seed(self.seed[iref][iseed])
                
                # Initialize batch of Gaussian random white initial maps for the synthesis
                imap = np.random.randn(self.NGEN, 2, 12*self.nside*self.nside)
 

                if self.ave_target:
                    # if average-target scale the intial white noise to match one of the input map std dev
                    ran_ind = np.random.randint(0, self.n_new_samples)
                    imap[:, 0] = imap[:, 0] * np.std(self.im[ran_ind, 0, :])
                    imap[:, 1] = imap[:, 1] * np.std(self.im[ran_ind, 1, :])

                else:
                    # Scale the random to match the current input map (ref) std dev
                    imap[:, 0] = imap[:, 0] * np.std(self.im[iref, 0, :])
                    imap[:, 1] = imap[:, 1] * np.std(self.im[iref, 1, :])

                # run syntesis using HealpixML
                omap = sy.run(
                    imap,
                    EVAL_FREQUENCY=10,
                    NUM_EPOCHS=self.nstep
                ).numpy()

                # Store best loss for each map in the batch
                floss[iseed] = np.min(sy.get_history())

                # Store results
                for k in range(iseed*self.NGEN, (iseed+1)*self.NGEN):
                    allmap[k] = omap[k-iseed*self.NGEN] * self.dim + self.mdim

                # Save partial results
                if first:
                    lim = self.im * self.dim + self.mdim
                    np.save(self.outpath + f'in_{f_outname}_map_{self.nside}.npy', lim)
                    if mask is not None:
                        np.save(self.outpath + f'mm_{f_outname}_map_{self.nside}.npy', mask[0])
                    np.save(self.outpath + f'out_{f_outname}_log_{self.nside}.npy', sy.get_history())
                    np.save(self.outpath + f'out_{f_outname}_map_{self.nside}.npy', 
                            omap[k-iseed*self.NGEN]*self.dim + self.mdim)
                    first=False

            # Final save
            np.save(self.outpath + f'out_{f_outname}_map_{self.nside}.npy', allmap)
            np.save(self.outpath + f'out_{f_outname}_loss_{self.nside}.npy', floss)
        
        print("[LOOP_SYNTHESIS] Computation Done.")


    # -------------------------------------------------------------------------
    # 8) Master run method calling sub-steps
    # -------------------------------------------------------------------------
    def run(self):
        """
        High-level method that calls each sub-step in order.
        """
        # 1) Preprocessing
        self.preprocess_data()

        if self.gauss_real:
            # 2) PCA-based Gaussian realization from the reference pixel covariance matrix
            self.generate_gaussian_maps()

        # 5) Compute reference scattering for each map
        self.init_reference_ps()

        # 3) Normalize data
        self.normalize_data()

        # 4) Initialize scattering operator and orientation matrices
        self.init_scat_and_orientation_matrices()

        # 5) Compute reference SC coefficients for each map in the input dataset
        self.init_reference_scat()

        # 6) Initialize batch of running maps to white noise and run synthesis loop
        self.loop_synthesis()

        print("[RUN] All steps completed.")
