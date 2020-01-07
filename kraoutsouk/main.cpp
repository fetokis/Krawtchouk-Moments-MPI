#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "moments.h"
#include <stdio.h>
#include <stdint.h>
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
	int myRank;
	int size;
	int x = 0, y = 0, p = 0, q = 0;
	double kpx, kqy, s1, s2, s3, s4, s5, s6,s7,s8, KMpq, fxy;
	double st4[100][100];
	double st5[63][128];
	double st3[128][128];
	double st2[63][128];
	double st1[128][128];
	double array[128][128];
	Order = 50;
	Mat rgb_image;
	rgb_image = imread("C:\\Users\\Kotzir\\Desktop\\kraoutsouk\\kraoutsouk\\kraoutsouk\\Lena_input.bmp", 0);// Lena64.bmp  Lena Image.bmp
	if (rgb_image.empty()) {
		cout << "Error : Image cannot be loaded..!!" << endl;
		exit(EXIT_FAILURE);
	}
	//Find width, height of image
	Size s = rgb_image.size();
	//Put on 2d array F[i][j] pixels from image lenas
	for (int i = 0; i < s.width; i++) {
		for (int j = 0; j < s.height; j++) {
			InputImage.F[i][j] = rgb_image.ptr<uchar>(j)[i];
		}
	}
	InputImage.ImWidth = s.width;
	InputImage.ImHeight = s.height;
	N = InputImage.ImWidth;
	M = InputImage.ImHeight;

	

	double p1 = 0.5;
	double p2 = 0.5;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


	/*if (myRank == 0) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				array[i][j] = InputImage.F[i][j];
				
				
			}
		}
	}
	MPI_Bcast(array, N*M, MPI_DOUBLE, 0, MPI_COMM_WORLD);*/
	if (myRank == 0) {
		Weight_Function(Wx, p1, N);
		Weight_Function(Wy, p2, M);
		for (p = 0; p <= Order; p++)
		{
			for (q = 0; q <= Order; q++)
			{
				s5 = 0;
				for (x = 0; x <=63; x++)
				{
					kpx = Krawtchouk_bar_poly_X(p, p1, x, N - 1, Wx);

					s6 = 0;
					for (y = 0; y < M; y++)
					{
						kqy = Krawtchouk_bar_poly_Y(q, p2, y, M - 1, Wy);

						fxy = (double)(InputImage.F[x][y]);
						//cout << "" << fxy;
						s6 = s6 + kqy*fxy;
						//cout << "" << s6;
					}
					s5 = s5 + kpx*s6;
					//cout << "" << s5;
				}


				KMpq = s5;
				K[p][q] = KMpq;

				//cout << "" << K[p][q];
			}
		}
		for (x = 0; x <=63; x++)
		{
			for (y = 0; y < M; y++)
			{
				s2 = 0;
				for (p = 0; p <= Order; p++)
				{
					kpx = Krawtchouk_bar_poly_X(p, p1, x, N - 1, Wx);
					s1 = 0;
					for (q = 0; q <= Order; q++)
					{
						kqy = Krawtchouk_bar_poly_Y(q, p2, y, M - 1, Wy);
						s1 = s1 + K[p][q] * kqy;
						//cout << "" << s1;

					}
					s2 = s2 + kpx*s1;
					//cout << "" << s2;
					st5[x][y] = s2;
					
					
					//cout << "" << st5[x][y];
				}

			}
		}
	}
	if (myRank == 1) {
		cout << "Hello" << endl;
		Weight_Function(Wx, p1, N);
		Weight_Function(Wy, p2, M);
		for (p = 0; p <= Order; p++)
		{
			for (q = 0; q <= Order; q++)
			{
				s5 = 0;
				for (x = 64; x < N; x++)
				{
					kpx = Krawtchouk_bar_poly_X(p, p1, x, N - 1, Wx);

					s6 = 0;
					for (y = 0; y < M; y++)
					{
						kqy = Krawtchouk_bar_poly_Y(q, p2, y, M - 1, Wy);

						fxy = (double)(InputImage.F[x][y]);
						//cout << "" << fxy;
						s6 = s6 + kqy*fxy;
						//cout << "" << s6;
					}
					s5 = s5 + kpx*s6;
					//cout << "" << s5;
				}cout << "" << s5;


				KMpq = s5;
				K1[p][q] = KMpq;

				//cout << "" << K1[p][q];
			}
		}
		for (x = 64; x < N; x++)
		{
			for (y = 0; y < M; y++)
			{
				s2 = 0;
				for (p = 0; p <= Order; p++)
				{
					kpx = Krawtchouk_bar_poly_X(p, p1, x, N - 1, Wx);
					s1 = 0;
					for (q = 0; q <= Order; q++)
					{
						kqy = Krawtchouk_bar_poly_Y(q, p2, y, M - 1, Wy);
						s1 = s1 + K1[p][q] * kqy;
						//cout << "" << s1;

					}
					s2 = s2 + kpx*s1;
					//cout << "" << s2;
					st2[x][y] = s2;
					
					
					//cout << "" << st2[x][y];
				}

			}
		}
	}
/*	for (int i = 0; i <=63; i++)
	{
		for (int j = 0; j < M; j++)
		{
			st1[i][j] = st5[i][j];
			st1[i+64][j] = st2[i][j];
		}
	}*/
	MPI_Finalize();
		Mat recImg(N, M, CV_8UC1);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {

				recImg.ptr<uchar>(j)[i] = (st5[i][j]);
				//st3[i][j] = recImg.ptr<uchar>(j)[i];
				//cout << "" << st3[i][j];
			}

		}
		Mat recIm(N, M, CV_8UC1);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {

				recIm.ptr<uchar>(j)[i] = (st2[i][j]);
				//st3[i][j] = recIm.ptr<uchar>(j)[i];
				//cout << "" << st3[i][j];
			}

		}
	
	
	
	imwrite("C:\\Users\\Kotzir\\Desktop\\Lena_output.bmp", recImg);
	imwrite("C:\\Users\\Kotzir\\Desktop\\Lena_output2.bmp", recIm);
	//MPI_Finalize();
	return 0;
	//1  factor calculation
}

double Factorial(int end) //factorial tou n! THREAD 1
{
	double result;
	int i;

	if (end == 0)
		return(1.0);

	result = 1;
	for (i = 1; i <= end; i++)
		result = result*i;

	return(result);
}


//2 calculation pochhammer

double pochhammer(int a, int k) //(a)^k = a(a+1)...(a+k+1) pochhammer symbol which you use in the norm and is calculated with the first type. THREAD 1
{
	double poch;
	int i;

	if (k == 0)
		poch = 1;
	else
	{
		poch = 1;
		for (i = 1; i <= k; i++)
			poch = poch*(a + i - 1);
	}
	return(poch);
}
//3 calculation norm

double p_norm(int n, double p, int N) // Ypologismos ths normas THREAD 1
{
	double pnorm;

	pnorm = pow(-1, double(n))*pow((1 - p) / p, n)*(Factorial(n) / pochhammer(-N, n));

	return(pnorm);
}

//weight calculation function



void Weight_Function(double *W, double p, int N) //weight calculation function THREAD 1
{
	int x;

	W[0] = pow(1 - p, N);
	for (x = 0; x <= N - 1; x++)
		W[x + 1] = ((double)(N - x) / (x + 1))*(double)(p / (1 - p))*(double)W[x];
}
void Weight_Function2(double *W1, double p, int N) //weight calculation function THREAD 1
{
	int x;

	W1[0] = pow(1 - p, N);
	for (x = 0; x <= N - 1; x++)
		W1[x + 1] = ((double)(N - x) / (x + 1))*(double)(p / (1 - p))*(double)W1[x];
}


//Calculation Krawtchouk for X

double Krawtchouk_bar_poly_X(int order, double p, int xy, int N, double *W) // THREAD 2
{
	double k[MAX_ORDER], A, B, w;
	int n,ni;

	w = W[xy];

	switch (order)
	{
	case 0:
		k[0] = sqrt(w / p_norm(0, p, N));
		break;
	case 1:
		k[1] = (1 - (xy / (p*N)))*sqrt(w / p_norm(1, p, N));
		break;
	default:
		k[0] = sqrt(w / p_norm(0, p, N));
		k[1] = (1 - (xy / (p*N)))*sqrt(w / p_norm(1, p, N));
		for (n = 1; n<order; n++)
		{

			A = (double)sqrt(p*(N - n) / (double)((1 - p)*(n + 1)));
			B = (double)sqrt((pow(p, 2)*(N - n)*(N - n + 1)) / (double)(pow(1 - p, 2)*(n + 1)*n));
			k[n + 1] = (A*(N*p - 2 * n*p + n - xy)*k[n] - B*n*(1 - p)*k[n - 1]) / (p*(N - n));
		}
		break;
	}
	return(k[order]);
}


//Calculation Krawtchouk for Y

double Krawtchouk_bar_poly_Y(int order, double p, int xy, int N, double *W) //THREAD 3
{
	double k[MAX_ORDER], A, B, w;
	int n,ni;

	w = W[xy];

	switch (order)
	{
	case 0:
		k[0] = sqrt(w / p_norm(0, p, N));
		break;
	case 1:
		k[1] = (1 - (xy / (p*N)))*sqrt(w / p_norm(1, p, N));
		break;
	default:
		k[0] = sqrt(w / p_norm(0, p, N));
		k[1] = (1 - (xy / (p*N)))*sqrt(w / p_norm(1, p, N));
		for (n = 1; n<order; n++)
		{

			A = (double)sqrt(p*(N - n) / (double)((1 - p)*(n + 1)));
			B = (double)sqrt((pow(p, 2)*(N - n)*(N - n + 1)) / (double)(pow(1 - p, 2)*(n + 1)*n));
			k[n + 1] = (A*(N*p - 2 * n*p + n - xy)*k[n] - B*n*(1 - p)*k[n - 1]) / (p*(N - n));
		}
		break;
	}
	return(k[order]);
}
