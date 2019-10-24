#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define min(a,b) ((a) <= (b) ? (a) : (b))

void computeNext(double** x0, double** x, double dt, double hx, double hy,
	double* diff, int me, int* xs, int* ys, int* xe, int* ye, double k0) {

	/* Index variables */
	int i, j;

	/* Factors for the stencil */
	double diagx, diagy, weightx, weighty;

	/* Local variable for computing difference */
	double ldiff;

	/*
	The stencil of the explicit operator for the heat equation
	on a regular rectangular grid using a five point finite difference
	scheme in space is :

	|                                    weightx * x[i-1][j]                                   |
	|                                                                                          |
	| weighty * x[i][j-1]   (diagx * weightx + diagy * weighty) * x[i][j]  weighty * x[i][j+1] |
	|                                                                                          |
	|                                    weightx * x[i+1][j]                                   | */

	diagx = -2.0 + hx*hx / (2 * k0*dt);
	diagy = -2.0 + hy*hy / (2 * k0*dt);
	weightx = k0*dt / (hx*hx);
	weighty = k0*dt / (hy*hy);

	/* Perform an explicit update on the points within the domain. */
	for (i = xs[me]; i <= xe[me]; i++)
		for (j = ys[me]; j <= ye[me]; j++)
			x[i][j] = weightx*(x0[i - 1][j] + x0[i + 1][j] + x0[i][j] * diagx)
			+ weighty*(x0[i][j - 1] + x0[i][j + 1] + x0[i][j] * diagy);

	/* Compute the difference into domain for convergence.
	Update the value x0(i,j). */
	*diff = 0.0;
	for (i = xs[me]; i <= xe[me]; i++)
		for (j = ys[me]; j <= ye[me]; j++) {
			ldiff = x0[i][j] - x[i][j];
			*diff += ldiff*ldiff;
			x0[i][j] = x[i][j];
		}
}


/* This subroutine sets up the initial temperatures on borders and inside */

void initValues(double** x0, int size_total_x, int size_total_y, double temp1_init, double temp2_init) {

	/* Index variables */
	int i, j;

	/* Setup temp1_init on borders */
	for (i = 0; i <= size_total_x - 1; i++) {
		x0[i][0] = temp1_init;
		x0[i][size_total_y - 1] = temp1_init;
	}

	for (j = 0; j <= size_total_y - 1; j++) {
		x0[0][j] = temp1_init;
		x0[size_total_x - 1][j] = temp1_init;
	}

	for (i = 0; i <= size_total_x - 2; i++) {
		x0[i][1] = temp1_init;
		x0[i][size_total_y - 2] = temp1_init;
	}

	for (j = 1; j <= size_total_y - 2; j++) {
		x0[1][j] = temp1_init;
		x0[size_total_x - 2][j] = temp1_init;
	}

	/* Setup temp2_init inside */
	for (i = 2; i <= size_total_x - 3; i++)
		for (j = 2; j <= size_total_y - 3; j++)
			x0[i][j] = temp2_init;
}

/*             Update Bounds of subdomain with me process          */
void updateBound(double ** x, int * neighBor, MPI_Comm comm2d, MPI_Datatype column_type, int me, int * xs, int * ys, int * xe, int * ye, int ycell)
{
	int S = 0, E = 1, N = 2, W = 3;
	int flag;
	MPI_Status status;

	/****************** North/South communication ******************/
	flag = 1;
	/* Send my boundary to North and receive from South */
	MPI_Sendrecv(&x[xe[me]][ys[me]], ycell, MPI_DOUBLE, neighBor[N], flag, &x[xs[me] - 1][ys[me]], ycell,
		MPI_DOUBLE, neighBor[S], flag, comm2d, &status);

	/* Send my boundary to South and receive from North */
	MPI_Sendrecv(&x[xs[me]][ys[me]], ycell, MPI_DOUBLE, neighBor[S], flag, &x[xe[me] + 1][ys[me]], ycell,
		MPI_DOUBLE, neighBor[N], flag, comm2d, &status);

	/****************** East/West communication ********************/
	flag = 2;
	/* Send my boundary to East and receive from West */
	MPI_Sendrecv(&x[xs[me]][ye[me]], 1, column_type, neighBor[E], flag, &x[xs[me]][ys[me] - 1], 1, column_type,
		neighBor[W], flag, comm2d, &status);

	/* Send my boundary to West and receive from East */
	MPI_Sendrecv(&x[xs[me]][ys[me]], 1, column_type, neighBor[W], flag, &x[xs[me]][ye[me] + 1], 1, column_type,
		neighBor[E], flag, comm2d, &status);
}

/* This subroutine computes the coordinates xs, xe, ys, ye, */
/* for each cell on the grid, respecting processes topology */

void processToMap(int *xs, int *ys, int *xe, int *ye, int xcell, int ycell, int x_domains, int y_domains) {

	/* Index variables */
	int i, j;

	/* Computation of starting ys,ye on (Ox) standard axis
	for the first column of global domain,
	Convention x(i,j) with i row and j column */
	for (i = 0; i<x_domains; i++) {
		ys[i] = 1;
		/* Here, ye(0:(x_domains-1)) = 1+ycell-1 */
		ye[i] = ys[i] + ycell - 1;
	}

	/* Computation of ys,ye on (Ox) standard axis
	for all other cells of global domain */
	for (i = 1; i<y_domains; i++)
		for (j = 0; j<x_domains; j++) {
			ys[i*x_domains + j] = ys[(i - 1)*x_domains + j] + ycell + 1;
			ye[i*x_domains + j] = ys[i*x_domains + j] + ycell - 1;
		}

	/* Computation of starting xs,xe on (Oy) standard axis
	for the first row of global domain,
	Convention x(i,j) with i row and j column */
	for (i = 0; i<y_domains; i++) {
		xs[i*x_domains] = 1;
		/* Here, xe(i*x_domains) = 1+xcell-1 */
		xe[i*x_domains] = xs[i*x_domains] + xcell - 1;
	}

	/* Computation of xs,xe on (Oy) standard axis
	for all other cells of global domain */
	for (i = 1; i <= y_domains; i++)
		for (j = 1; j<x_domains; j++) {
			xs[(i - 1)*x_domains + j] = xs[(i - 1)*x_domains + (j - 1)] + xcell + 1;
			xe[(i - 1)*x_domains + j] = xs[(i - 1)*x_domains + j] + xcell - 1;
		}
}


int main(int argc, char **argv)
{
	MPI_Comm comm, comm2d;
	int dims[2];
	int periods[2];
	int reorganisation = 0;
	int ndims, nproc, rank, maxStep;

	int S = 0, E = 1, N = 2, W = 3;
	int neighBor[4];
	int xcell, ycell;
	int *xs, *ys, *xe, *ye;

	MPI_Datatype column_type;

	int size_x, size_y, x_domains, y_domains;
	int size_total_x, size_total_y;

	double dt, result, epsilon, dt1, dt2, hx, hy;
	double localDiff;
	double temp1_init, temp2_init, k0;

	int convergence = 0;

	/* Time and step variables */
	double t;
	int step;

	/* Variables for clock */
	double time_init, time_final, elapsed_time;

	/* Arrays */
	double **x;
	double **x0;
	double *xtemp;
	double *xfinal;

	/* Index variables */
	int i, j, k, l;

	FILE* file;

	MPI_Init(&argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);


	size_x = 256;
	size_y = 128;
	x_domains = /*8*/atoi(argv[1]);
	y_domains = /*4*/ atoi(argv[2]);
	maxStep = 100000;
	dt1 = 1.0e-1;
	epsilon = 1.0e-1;

	k0 = 1;

	hx = 1.0 / (double)(size_x);
	hy = 1.0 / (double)(size_y);
	dt2 = 0.25*(min(hx, hy)*min(hx, hy)) / k0;
	size_total_x = size_x + 2 * x_domains;
	size_total_y = size_y + 2 * y_domains;

	/* Take a right time step for convergence */
	if (dt1 >= dt2)
		dt = dt2;
	else 
		dt = dt1;

	/* Allocate final 1D array */
	xfinal = malloc(size_x*size_y * sizeof(*xfinal));

	/* Allocate 2D contiguous arrays x and x0 */
	/* Allocate size_total_x rows */
	x = malloc(size_total_x * sizeof(*x));
	x0 = malloc(size_total_x * sizeof(*x0));
	/* Allocate x[0] and x0[0] for contiguous arrays */
	x[0] = malloc(size_total_x*size_total_y * sizeof(**x));
	x0[0] = malloc(size_total_x*size_total_y * sizeof(**x0));
	/* Loop on rows */
	for (i = 1; i<size_total_x; i++) {
		/* Increment size_total_x block on x[i] and x0[i] address */
		x[i] = x[0] + i*size_total_y;
		x0[i] = x0[0] + i*size_total_y;
	}

	/* Allocate coordinates of processes */
	xs = malloc(nproc * sizeof(int));
	xe = malloc(nproc * sizeof(int));
	ys = malloc(nproc * sizeof(int));
	ye = malloc(nproc * sizeof(int));

	/* Create 2D cartesian grid */
	periods[0] = 0;
	periods[1] = 0;
	
	ndims = 2;
	
	dims[0] = y_domains;
	dims[1] = x_domains;
	MPI_Cart_create(comm, ndims, dims, periods, reorganisation, &comm2d);

	/* Identify neighBors */
	neighBor[0] = MPI_PROC_NULL;
	neighBor[1] = MPI_PROC_NULL;
	neighBor[2] = MPI_PROC_NULL;
	neighBor[3] = MPI_PROC_NULL;

	/* Left/West and Right/East neighBors */
	MPI_Cart_shift(comm2d, 0, 1, &neighBor[W], &neighBor[E]);

	/* Bottom/South and Upper/North neighBors */
	MPI_Cart_shift(comm2d, 1, 1, &neighBor[S], &neighBor[N]);

	/* Size of each cell */
	xcell = (size_x / x_domains);
	ycell = (size_y / y_domains);

	/* Allocate subdomain */
	xtemp = malloc(xcell*ycell * sizeof(*xtemp));

	/* Compute xs, xe, ys, ye for each cell on the grid */
	processToMap(xs, ys, xe, ye, xcell, ycell, x_domains, y_domains);

	/* Create column data type to communicate with East and West neighBors */
	MPI_Type_vector(xcell, 1, size_total_y, MPI_DOUBLE, &column_type);
	MPI_Type_commit(&column_type);

	/* temp1_init: temperature init on borders */
	temp1_init = 10.0;

	/* temp2_init: temperature init inside */
	temp2_init = -10.0;


	/* Initialize values */
	initValues(x0, size_total_x, size_total_y, temp1_init, temp2_init);

	if (rank == 0)
	{
		/*Print values*/
		file = fopen("D:\\godina\\Paralelni sistemi\\Lab vezbe\\Zadatak\\input.txt", "w");
		fprintf(file, "Starting values:");
		fprintf(file, "\n");
		for (i = 0; i < size_total_x; i++)
		{
			for (j = 0; j < size_total_y; j++)
			{
				fprintf(file, "%5.2f ", x0[i][j]);
				fprintf(file, " ");
			}

			fprintf(file, "\n");
		}
		fclose(file);

		printf("\n  Starting values are in input.txt\n");

	/*	printf("Starting values:");
		printf("\n");
		for (i = 0; i < size_total_x; i++)
		{
			for (j = 0; j < size_total_y; j++)
				printf("%5.2f ", x0[i][j]);

			printf("\n");
		}*/
	}

	/* Update the boundaries */
	updateBound(x0, neighBor, comm2d, column_type, rank, xs, ys, xe, ye, ycell);

	/* Initialize step and time */
	step = 0;
	t = 0.0;

	/* Starting time */
	time_init = MPI_Wtime();

	/* Main loop : until convergence */
	while (!convergence) {
		/* Increment step and time */
		step = step + 1;
		t = t + dt;
		/* Perform one step of the explicit scheme */
		computeNext(x0, x, dt, hx, hy, &localDiff, rank, xs, ys, xe, ye, k0);
		/* Update the partial solution along the interface */
		updateBound(x0, neighBor, comm2d, column_type, rank, xs, ys, xe, ye, ycell);
		/* Sum reduction to get global difference */
		MPI_Allreduce(&localDiff, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
		/* Current global difference with convergence */
		result = sqrt(result);
		/* Break if convergence reached or step greater than maxStep */
		if ((result<epsilon) || (step>maxStep)) break;
	}

	/* Ending time */
	time_final = MPI_Wtime();
	/* Elapsed time */
	elapsed_time = time_final - time_init;

	/* Gather all subdomains :
	inner loop on columns index (second index)
	to optimize since C is row major */
	j = 1;
	for (i = xs[rank]; i <= xe[rank]; i++) {
		for (k = 0; k<ycell; k++)
			xtemp[(j - 1)*ycell + k] = x0[i][ys[rank] + k];
		j = j + 1;
	}

	/* Perform gathering */
	MPI_Gather(xtemp, xcell*ycell, MPI_DOUBLE, xfinal, xcell*ycell, MPI_DOUBLE, 0, comm);

	/* Print results */
	if (rank == 0) {
		printf("\n");
		printf("  Time step = %.9e\n", dt);
		printf("\n");
		printf("  Convergence = %.9f after %d steps\n", result, step);
		printf("\n");
		printf("  Problem size = %d\n", size_x*size_y);
		printf("\n");
		printf("  Elapsed time = %.9f\n", elapsed_time);
		printf("\n");

		/*Print values*/
		file = fopen("D:\\godina\\Paralelni sistemi\\Lab vezbe\\Zadatak\\output.txt", "w");
		fprintf(file, "Ending values:");
		fprintf(file, "\n");
		for (i = 0; i < size_x; i++)
		{
			for (j = 0; j < size_y; j++)
				fprintf(file, "%5.2f ", xfinal[i*size_y + j]);

			fprintf(file, "\n");
		}
		fclose(file);

	/*	printf("Ending values:");
		printf("\n");
		for (i = 0; i < size_x; i++)
		{
			for (j = 0; j < size_y; j++)
				printf("%5.2f ", xfinal[i*size_y + j]);

			printf("\n");
		}*/

		printf("  Computed solution in output.txt\n");
	}

	/* Free all arrays */
	free(x[0]);
	free(x);
	free(x0[0]);
	free(x0);
	free(xtemp);
	free(xfinal);
	free(xs);
	free(ys);
	free(xe);
	free(ye);

	/* Free column type */
	MPI_Type_free(&column_type);

	/* Finish MPI */
	MPI_Finalize();

	return 0;

}