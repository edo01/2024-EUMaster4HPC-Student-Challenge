#include "ConjugateGradient_MultiGPUS_CUDA_NCCL.cuh"

namespace LAM
{

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::solve(int max_iters, FloatingType rel_error)
    {
        //initializing MPI
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        // since only the matrix-vector multiplication is parallelized,
        // these variables are only needed in the device 0 of rank 0
        FloatingType * b_dev, * x_dev, * alpha_dev, * beta_dev, * bb_dev;
        FloatingType * rr_dev, * rr_new_dev, * r_dev, * pAp_dev, * Ap0_dev;

        // const to store on the device 
        //FloatingType * const_uno_dev, * const_zero_dev;
        FloatingType const_uno_dev=1, const_zero_dev=0;

        // module of the rhs and of the residual on the host
        FloatingType * rr, * bb;
        FloatingType * dot_res, *alpha_host, *beta_host;


        bool stop = false;
        cudaHostAlloc(&rr, sizeof(FloatingType), cudaHostAllocDefault);
        cudaHostAlloc(&bb, sizeof(FloatingType), cudaHostAllocDefault);
        int num_iters=1;

        // variables on the host
        cudaHostAlloc(&dot_res, sizeof(FloatingType), cudaHostAllocDefault);
        cudaHostAlloc(&alpha_host, sizeof(FloatingType), cudaHostAllocDefault);
        cudaHostAlloc(&beta_host, sizeof(FloatingType), cudaHostAllocDefault);

        /*
        *  ---------------------------------------------------------------
        *  -----------------  Device memory allocation  ------------------
        *  ---------------------------------------------------------------
        */
        /* Allocate and initialize a given number of rows of A to each device
           data for computating the matrix-vector multiplication */
        
        // stores the pointer to the result of the matrix-vector multiplication of the i-th device
        FloatingType * Ap_dev[_numDevices];

        /* stores the pointer to the vector for the matrix-vector multiplication of the i-th device
           of the i-th device */
        FloatingType * p_dev[_numDevices];

        // Allocate p and Ap on all devices
        #pragma omp parallel for num_threads(_numDevices)
        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
 
            cudaMalloc(&Ap_dev[i], _rows_per_device[i] * sizeof(FloatingType));
            cudaMalloc(&p_dev[i], _num_cols * sizeof(FloatingType));
        }

        // Allocate the in device 0 of rank 0 the other variables
        if (rank == 0) {
            printf("Allocating memory on the device 0 of rank 0\n");
            fflush(stdout);
            cudaSetDevice(0);
            //allocate the rhs and the result of the CG method
            cudaMalloc(&b_dev, sizeof(FloatingType) * _num_cols);
            cudaMalloc(&x_dev, sizeof(FloatingType) * _num_cols);
            //allocate alpha and beta on the device
            cudaMalloc(&alpha_dev, sizeof(FloatingType));
            cudaMalloc(&beta_dev, sizeof(FloatingType));
            //allocate the module of the rhs and the residual
            cudaMalloc(&bb_dev, sizeof(FloatingType));
            cudaMalloc(&rr_dev, sizeof(FloatingType));
            cudaMalloc(&rr_new_dev, sizeof(FloatingType));
            cudaMalloc(&r_dev, sizeof(FloatingType) * _num_cols);
            cudaMalloc(&pAp_dev, sizeof(FloatingType));

            // Ap0_dev is located in device 0 and will collect all the result from the devices
            cudaMalloc(&Ap0_dev, sizeof(FloatingType) * _num_cols); 

            // constant set on the device
            //cudaMallocHost(&const_uno_dev, sizeof(FloatingType));
            //cudaMallocHost(&const_zero_dev, sizeof(FloatingType));
            /*cudaMalloc(&const_uno_dev, sizeof(FloatingType));
            cudaMalloc(&const_zero_dev, sizeof(FloatingType));
            FloatingType uno = 1;
            FloatingType zero = 0;
            cudaMemcpy(const_uno_dev, &uno, sizeof(FloatingType), cudaMemcpyHostToDevice);
            cudaMemcpy(const_zero_dev, &zero, sizeof(FloatingType), cudaMemcpyHostToDevice);*/

            // copy rhs to device
            cudaMemcpy(b_dev, _rhs, sizeof(FloatingType) * _num_cols, cudaMemcpyHostToDevice);
            // set the initial value of x to 0
            cudaMemset(x_dev, 0, sizeof(FloatingType) * _num_cols);
            // set the initial value of r to b
            cudaMemcpy(r_dev, b_dev, sizeof(FloatingType) * _num_cols, cudaMemcpyDeviceToDevice); 
            // set the initial value of p to b
            cudaMemcpy(p_dev[0], b_dev, sizeof(FloatingType) * _num_cols, cudaMemcpyDeviceToDevice); 

            // compute the module of the rhs
            cublasStatus_t status = cublasDdot(cublas_handler[0], _num_cols, b_dev, 1, b_dev, 1, dot_res);
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("cublasDot failed\n");
                fflush(stdout);
                return false;
            }
            printf("bb: %f\n", *dot_res);
            fflush(stdout);

            // since dot_res is a pointer to the result of the dot product in the HOST, it is necessary to copy the
            // result back to the host
            cudaMemcpy(bb_dev, dot_res, sizeof(FloatingType), cudaMemcpyDeviceToDevice);
            *bb = *dot_res;
            //dot<FloatingType>(b_dev, b_dev, bb_dev, _num_cols, streams[0]);

            // set the initial value of rr to bb
            cudaMemcpy(rr_dev, dot_res, sizeof(FloatingType), cudaMemcpyDeviceToDevice);
            *rr = *dot_res;

            printf("initialization done\n");
        }

        /*
        *  ---------------------------------------------------------------
        *  -----------------  NCCL initialization  -----------------------
        *  ---------------------------------------------------------------
        */
        int tot_num_devices = _numDevices * num_procs;
        
        // id is used to identify a NCCL communication group
        ncclUniqueId id;
        
        /*
         * Each ncclComm_t is associated with a specific GPU device because NCCL
         * is designed to perform collective communication operations among a group 
         * of GPUs. Each GPU in the group needs to have its own unique communicator 
         * to manage its part of the collective operation.
        */
        ncclComm_t comms[_numDevices];
        
        // Generating NCCL unique ID at one process and broadcasting it to all
        if (rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        //  Initializing NCCL
        ncclGroupStart();
        for (int i=0; i<_numDevices; i++) {
            cudaSetDevice(i);
            int device_rank = rank * _numDevices + i;

            // ncclCommInitRank is used to create a new communicator for each device
            // it takes the communicator, the number of devices in the NCCL group,
            // the unique ID of the NCCL communication group, the rank of
            // the devices in the NCCL group
            ncclCommInitRank(&comms[i], tot_num_devices, id, device_rank);
        }
        ncclGroupEnd();

        /*
        *  ---------------------------------------------------------------
        *  -----------------  CG Algorithm  ------------------------------
        *  ---------------------------------------------------------------
        */ 

        // CG Iterations
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {

            printf("Broadcast p\n");
            fflush(stdout);

            // Copy p from device 0 to all other devices to allow matrix-vector multiplication
            ncclGroupStart();
            for (int i=0; i < _numDevices; i++) {
                ncclBroadcast(p_dev[0], p_dev[i], _num_cols, nccl_datatype, 0, comms[i], streams[i]);
            }
            ncclGroupEnd();

            // Synchronizing on CUDA stream to complete NCCL communication
            for (int i = 0; i < _numDevices; i++) {
                cudaStreamSynchronize(streams[i]);
            }

            // Performs matrix-vector multiplication in each device of each rank
            //#pragma omp parallel for num_threads(_numDevices)
            for(int i = 0; i < _numDevices; i++){
                cudaSetDevice(i);
                printf("Device %d: matrix-vector multiplication\n", i);
                printf("p_dev[0]: ");

                FloatingType * p_host = (FloatingType *)malloc(sizeof(FloatingType) * _num_cols);
                cudaMemcpy(p_host, p_dev[i], sizeof(FloatingType) * _num_cols, cudaMemcpyDeviceToHost);
                for(int j = 0; j < _num_cols; j++){
                    printf("%f ", p_host[j]);
                }
                fflush(stdout);
                free(p_host);

                //print rows_per_device
                printf("rows_per_device: %lu\n", _rows_per_device[i]);
                fflush(stdout);
                //print number of columns
                printf("number of columns: %lu\n", _num_cols);
                fflush(stdout);

                //print A_dev[i]
                FloatingType * A_host = (FloatingType *)malloc(sizeof(FloatingType) * _num_cols * _rows_per_device[i]);
                cudaMemcpy(A_host, _A_dev[i], sizeof(FloatingType) * _num_cols * _rows_per_device[i], cudaMemcpyDeviceToHost);
                for(int j = 0; j < _rows_per_device[i]; j++){
                    for(int k = 0; k < _num_cols; k++){
                        printf("%f ", A_host[j * _num_cols + k]);
                    }
                    printf("\n");
                }
                fflush(stdout);
                free(A_host);

                // check cublasDgemv
                cudaDeviceSynchronize();
                cublasStatus_t status = cublasDgemv(cublas_handler[i], CUBLAS_OP_N, _rows_per_device[i], _num_cols,
                    &const_uno_dev, _A_dev[i], _rows_per_device[i]*_num_cols, p_dev[i], 1, &const_zero_dev, Ap_dev[i], 1);

                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("cublasDgemv failed\n");
                    fflush(stdout);
                    return false;
                }

                cudaDeviceSynchronize();
                
                printf("Device %d: matrix-vector multiplication done\n", i);
                fflush(stdout);
            }
            
            // All-To-One Gather to collect all the results of the mat-vec multiplication in device 0 in rank 0
            ncclGroupStart();
            if(rank == 0) {
                int offset = 0;
                // for each device, the rank 0 collects the result of the mat-vec multiplication with a recv
                for(int i = 0; i < _numDevices * num_procs; i++){
                    if(i < _numDevices * (num_procs - 1)) {
                        // devices of rank 0 to num_procs - 2 have the same number of rows and so the same distribution among devices
                        ncclRecv(Ap0_dev + offset, _rows_per_device[i % _numDevices], nccl_datatype, i, comms[0], streams[0]);
                        offset += _rows_per_device[i % _numDevices];
                    } else {
                        // the last rank can have a different number of rows
                        unsigned int numRowsLastRank = _num_cols / num_procs + _num_cols % num_procs;
                        unsigned int numRowsDeviceLastRank = numRowsLastRank / _numDevices;
                        if (i == _numDevices * num_procs - 1){
                            // the last device of the last rank has the remaining rows
                            numRowsDeviceLastRank += numRowsLastRank % _numDevices;
                        }
                        ncclRecv(Ap0_dev + offset, numRowsDeviceLastRank, nccl_datatype, i, comms[0], streams[0]);
                        offset += numRowsDeviceLastRank;
                    }
                }
            }
            for(int i = 0; i < _numDevices; i++) {
                ncclSend(Ap_dev[i], _rows_per_device[i] , nccl_datatype, 0, comms[i], streams[i]);
            }
            ncclGroupEnd();

            // Synchronizing on CUDA stream to complete NCCL communication
            for (int i = 0; i < _numDevices; i++) {
                cudaStreamSynchronize(streams[i]);
            }

            // Device 0 in rank 0 carries on all the other operation involved in the iteration of the CG method
            if(rank == 0) {

                cudaSetDevice(0);

                //dot<FloatingType>(p_dev[0], Ap0_dev, pAp_dev, _num_cols, streams[0]);

                    // pAp=p*Ap
                    cublasDdot(cublas_handler[0], _num_cols, p_dev[0], 1, Ap0_dev, 1, dot_res);
                    cudaMemcpy(pAp_dev, dot_res, sizeof(FloatingType), cudaMemcpyHostToDevice);

                    //calculate alpha_dev= rr/pAp
                    *alpha_host = *rr/(*dot_res);
                    cudaMemcpy(alpha_dev, dot_res, sizeof(FloatingType), cudaMemcpyHostToDevice);

                    //calculate x=alpha*p+x
                    cublasDaxpy(cublas_handler[0], _num_cols, alpha_dev, p_dev[0], 1, x_dev, 1);
                
                    //calculate r=-alpha*Ap+r
                    *alpha_host = -*alpha_host;
                    cudaMemcpy(alpha_dev, &alpha_host, sizeof(FloatingType), cudaMemcpyHostToDevice);
                    cublasDaxpy(cublas_handler[0], _num_cols, alpha_dev, Ap0_dev, 1, r_dev, 1);

                    //calculate rr_new=r*r
                    cublasDdot(cublas_handler[0], _num_cols, r_dev, 1, r_dev, 1, dot_res);
                    cudaMemcpy(rr_new_dev, dot_res, sizeof(FloatingType), cudaMemcpyHostToDevice);

                    //calculate beta=rr_new/rr
                    *beta_host = *dot_res/(*rr);
                    cudaMemcpy(beta_dev, beta_host, sizeof(FloatingType), cudaMemcpyHostToDevice);

                    // rr=rr_new
                    *rr = *dot_res;

                

                cudaDeviceSynchronize();

                if (std::sqrt(*rr / *bb) < rel_error) { stop = true; }

                PRINT_RANK0("Iteration %d, relative error is %e\n", num_iters, std::sqrt(*rr / *bb));
                fflush(stdout);
            }

            // Rank 0 broadcasts the flag stop to all other rank in order to stop the computation when the solution is found
            MPI_Bcast(&stop, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            if(stop) { 
                PRINT_RANK0("Broadcast stop\n");    
                fflush(stdout);
                break;
            }


            // Device 0 in rank 0 computes the new value of p that will be broadcast to all other devices in the next iteration
            if (rank == 0){
                cudaSetDevice(0);
                //calculate p=beta*p+r
                cublasDaxpy(cublas_handler[0], _num_cols, beta_dev, p_dev[0], 1, r_dev, 1);
            }
        }

        /*
         * ---------------------------------------------------------------
         * ------------- Freeing memory and save the solution ------------
         * ---------------------------------------------------------------
         */

        // Device 0 of rank 0 prints the information about the result of the CG method
        if(rank == 0) {
            //cudaSetDevice(0);
            //cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            //cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);

            // Prints the number of iterations and the relative error
            if (num_iters <= max_iters) {
                printf("PARALLEL MULTI-GPUS CUDA NCCL: Converged in %d iterations, relative error is %e\n", num_iters,
                       std::sqrt(*rr / *bb));
            } else {
                printf("PARALLEL MULTI-GPUS CUDA NCCL: Did not converge in %d iterations, relative error is %e\n", max_iters,
                       std::sqrt(*rr / *bb));
            }
            fflush(stdout);

            // Copy solution to host
            cudaSetDevice(0);
            cudaMemcpy(_x, x_dev, _num_cols * sizeof(FloatingType), cudaMemcpyDeviceToHost);
            printf("moved x to host\n");
            fflush(stdout);

            // Free GPU memory
            cudaFree(alpha_dev);
            cudaFree(beta_dev);
            cudaFree(bb_dev);
            cudaFree(rr_dev);
            cudaFree(rr_new_dev);
            cudaFree(r_dev);
            cudaFree(p_dev);
            cudaFree(Ap0_dev);
            cudaFree(pAp_dev);
            
            printf("free GPU memory of rank 0\n");
            fflush(stdout);

            // Free CPU memory
            cudaFreeHost(dot_res);
            cudaFreeHost(alpha_host);
            cudaFreeHost(beta_host);
            cudaFreeHost(rr);
            cudaFreeHost(bb);
        }

        // All devices free their allocated memory and destroy streams
        //#pragma omp parallel for num_threads(_numDevices)
        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaFree(Ap_dev[i]);
            cudaFree(p_dev[i]);
        }

        printf("free GPU memory of other ranks\n");
        fflush(stdout);

        // Finalizing NCCL
        for (int i = 0; i < _numDevices; i++) {
            ncclCommDestroy(comms[i]);
        }

        printf("NCCL destroyed\n");
        fflush(stdout);

        return (num_iters <= max_iters);
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::load_matrix_from_file(const char * filename)
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        MPI_File fhandle;
        size_t num_total_rows;
        size_t file_offset;
        size_t numRowsPerDevice;
        

        // at the index i, it contains a portion of the matrix that will be transfered to the i-th device
        // it is just a temporary variable to store the portion of the matrix that will be transfered
        FloatingType** h_A;

        // Initialize an MPI file handler and open the file
        if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle) != MPI_SUCCESS) {
            printf("[MPI process %d] Failure in opening the file.\n", rank);
            //MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return false;
        }
        printf("[MPI process %d] File opened successfully.\n", rank);
        fflush(stdout);

        // Read from file the dimensions of the matrix
        MPI_File_read(fhandle, &num_total_rows, 1, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_File_read(fhandle, &_num_cols, 1, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);

        PRINT_RANK0("Number of rows: %lu\n", num_total_rows);
        PRINT_RANK0("Number of columns: %lu\n", _num_cols); 
        fflush(stdout);

        // Evaluate the number of rows associated to each rank and the offset in file
        _num_local_rows = num_total_rows / num_procs;
        file_offset = _num_local_rows * sizeof(FloatingType) * rank * _num_cols;
        _offset = _num_local_rows * rank;

        //the last rank will have the remaining rows
        if(rank == num_procs - 1){
            //add the reminder to the last rank
            _num_local_rows += num_total_rows % num_procs; 
        }

        // File pointer is set to the current pointer position plus offset in order to read the right portion of the matrix
        MPI_File_seek(fhandle, file_offset, MPI_SEEK_CUR);
        
        PRINT_RANK0("Number of local rows: %lu\n", _num_local_rows);
        PRINT_RANK0("Offset: %lu\n", _offset);
        fflush(stdout);

        // Allocates page-locked memory on the host for asynchronous memory copy
        cudaHostAlloc(&_rows_per_device, sizeof(size_t) * _numDevices, cudaHostAllocDefault);

        PRINT_RANK0("Allocated page-locked memory for the host\n");
        fflush(stdout);

        // Evaluate the number of rows associated to each device in the rank
        numRowsPerDevice = _num_local_rows / _numDevices;
        size_t s = 0;
        for(int i = 0; i < _numDevices; i++){
            // The last device will have the remaining rows
            _rows_per_device[i] = (s + numRowsPerDevice <= _num_local_rows) ? numRowsPerDevice : _num_local_rows - s;
            s += numRowsPerDevice;
        }
        if(s < _num_local_rows) _rows_per_device[_numDevices - 1] += _num_local_rows - s;

        // Allocate the space in each device for its chunk of the matrix
        #pragma omp parallel for num_threads(_numDevices)
        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaMalloc(&_A_dev[i], sizeof(FloatingType) * _num_cols * _rows_per_device[i]);
            PRINT_RANK0("Allocated space for the matrix in device %d\n", i);
            fflush(stdout);
        }



        // Read matrix from file and copy it into the devices
        cudaHostAlloc(&h_A, sizeof(FloatingType *) * _numDevices, cudaHostAllocDefault);
        /* for each device, allocate the space in the host and read the portion of the matrix from the file
           then copy it into the device*/ 
        for(int k = 0; k < _numDevices; k++) {
            cudaHostAlloc(&h_A[k], sizeof(FloatingType) * _num_cols * _rows_per_device[k], cudaHostAllocDefault);
            for (int i = 0; i < _rows_per_device[k]; i++) {
                for (int j = 0; j < _num_cols; j++){
                    MPI_File_read(fhandle, &h_A[k][i * (_num_cols) + j], 1, get_mpi_datatype(), MPI_STATUS_IGNORE);
                }
            }
            PRINT_RANK0("Read the portion %d of the matrix from the file\n", k);
            fflush(stdout);
            cudaSetDevice(k);
            // I'm pretty sure that these copies are done in serial, so it is not efficient 
            cudaMemcpy(_A_dev[k], h_A[k], sizeof(FloatingType) * _num_cols * _rows_per_device[k], cudaMemcpyHostToDevice);
        }

        // Close the file
        if(MPI_File_close(&fhandle) != MPI_SUCCESS) {
            printf("[MPI process %d] Failure in closing the file.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return false;
        }
        printf("[MPI process %d] File closed successfully.\n", rank);
        fflush(stdout);

        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaFreeHost(h_A[i]);
        }

        cudaFreeHost(h_A);
        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::load_rhs_from_file(const char * filename)
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        if(rank!=0) return true;  // Only rank 0 reads the right-hand side vector from the matrix

        FILE * file = fopen(filename, "rb");
        if(file == nullptr)
        {
            fprintf(stderr, "Cannot open output file\n");
            return false;
        }

        size_t rhs_rows, rhs_cols;
        fread(&rhs_rows, sizeof(size_t), 1, file);
        fread(&rhs_cols, sizeof(size_t), 1, file);

        if(rhs_cols != 1){
            fprintf(stderr, "The file does not contain a valid rhs\n");
            return false;
        }
        if(rhs_rows != _num_cols)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return false;
        }

        cudaHostAlloc(&_rhs, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        cudaHostAlloc(&_x, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        

        fread(_rhs, sizeof(FloatingType), rhs_rows, file);

        fclose(file);

        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::save_result_to_file(const char * filename) const
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
     
        if(rank!=0) return true;  // Only rank 0 writes the result to the file
     
        FILE * file = fopen(filename, "wb");
        if(file == nullptr)
        {
            fprintf(stderr, "Cannot open output file\n");
            return false;
        }
        int num_rows = _num_cols;
        int num_cols = 1;
        fwrite(&num_rows, sizeof(size_t), 1, file);
        fwrite(&num_cols, sizeof(size_t), 1, file);
        //save rhs to file
        fwrite(_x, sizeof(FloatingType), num_rows, file);

        fclose(file);

        return true;
    }



//template class LAM::ConjugateGradient_MultiGPUS_CUDA_NCCL<float>;
template class LAM::ConjugateGradient_MultiGPUS_CUDA_NCCL<double>;

}