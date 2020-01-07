#define MAX_WIDTH_IMSIZE    256
#define MAX_HEIGHT_IMSIZE   MAX_WIDTH_IMSIZE
#define MAX_ORDER           100


typedef struct
{
	int   F[MAX_WIDTH_IMSIZE][MAX_HEIGHT_IMSIZE];
	int   ImWidth;
	int   ImHeight;
}Image;

//Global Declarations
Image InputImage, RecImage;
int Order, N, M;
double K[MAX_ORDER][MAX_ORDER];
double K1[MAX_ORDER][MAX_ORDER];


//Function Prototypes
void Weight_Function(double *W, double p, int N);
void Weight_Function2(double *W1, double p, int N);

double pochhammer(int a, int k);
double p_norm(int n, double p, int N);
double Krawtchouk_bar_poly_X(int order, double p, int xy, int N, double *W);
double Krawtchouk_bar_poly_Y(int order, double p, int xy, int N, double *W);
double Wx[MAX_WIDTH_IMSIZE], Wy[MAX_WIDTH_IMSIZE];
double Wx1[MAX_WIDTH_IMSIZE], Wy1[MAX_WIDTH_IMSIZE];
double malloc2ddouble(double ***array, int n, int m);
