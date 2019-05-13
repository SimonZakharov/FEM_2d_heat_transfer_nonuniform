/*
	Row Representation Complete and Ordered
	Разреженный строчный формат представления матрицы
	Захаров С.А. 15-08-2017
*/
#define _CRT_SECURE_NO_WARNINGS

#ifndef MAX_ITER
#define MAX_ITER 20000
#endif // !MAX_ITER

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

const double EPS = 1e-16;
const double C	 = 1e50;

struct mat_el
{
	double value;
	int column;

	mat_el(double v, int c)
	{
		value = v;
		column = c;
	}
};

//	разреженный строчный формат представления матрицы
struct rrco
{
	//	количество уравнений в системе
	int N;
	//	элементы матрицы по строкам
	vector<vector<mat_el>> matrix;

	rrco()
	{
		N = 0;
		matrix.clear();
	}

	//	выделение памяти
	void memset(int n)
	{
		N = n;
		matrix.resize(n);
		for (int i = 0; i < n; i++)
			matrix[i].clear();
	}

	//	очистка памяти
	void clear()
	{
		for (int i = 0; i < N; i++)
		{
			matrix[i].clear();
		}
		matrix.clear();
	}

	double at(int row, int column)
	{
		for (int i = 0; i < matrix[row].size(); i++)
			if (column == matrix[row].at(i).column)
				return matrix[row].at(i).value;
		return NULL;
	}

	void add(double value, int row, int column)
	{
		for (int i = 0; i < matrix[row].size(); i++)
			if (column == matrix[row].at(i).column)
			{
				matrix[row].at(i).value += value;
				return;
			}
		matrix[row].push_back(mat_el(value, column));
	}

	//	вспомогательная функция - чтение из файла с матрицей, записанной в обычном виде
	void get_from_file(FILE *input)
	{
		if (input == NULL)
		{
			printf("Cannot open the input file to read the system\n");
			return;
		}
		fscanf(input, "%d", &N);
		matrix.resize(N);
		double v = 0.;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				fscanf(input, "%lf", &v);
				if (v != 0.)
					add(v, i, j);
			}
		}
	}

	//	вспомогательная функция - вывод матрицы в файл в обычном виде
	void print_to_file(FILE *output)
	{
		if (output == NULL)
		{
			printf("Cannot open the output file to print the system\n");
			return;
		}
		double v = 0.;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				v = at(i, j);
				if (v != 0.)
					fprintf(output, "%4.2lf\t", v);
				else
					fprintf(output, "%4.2lf\t", 0.);
			}
			fprintf(output, "\n");
		}
	}
};

struct input_data
{
	double x0, y0, xF, yF, srcX, srcY;
	int n_sigma;
	double *y_sigma;
	double stepX, stepY, coeffX, coeffY;

	input_data()
	{
		x0 = y0 = xF = yF = srcX = srcY = 0.;
		n_sigma = 0;
		y_sigma = NULL;
		stepX = stepY = 0.;
		coeffX = coeffY = 0.;
	}
};

//	чтение входных данных из файла в определенном формате
void read_input_data(FILE *input, input_data &dat)
{
	if (input == NULL)
	{
		printf("Cannot open the input file for reading\n");
		return;
	}
	fscanf(input, "%lf %lf", &dat.x0, &dat.y0);
	fscanf(input, "%lf %lf", &dat.xF, &dat.yF);
	fscanf(input, "%lf %lf", &dat.srcX, &dat.srcY);
	fscanf(input, "%d", &dat.n_sigma);
	dat.y_sigma = new double[dat.n_sigma];
	for (int i = 0; i < dat.n_sigma; i++)
		fscanf(input, "%lf", &dat.y_sigma[i]);
	fscanf(input, "%lf %lf", &dat.stepX, &dat.stepY);
	fscanf(input, "%lf %lf", &dat.coeffX, &dat.coeffY);
}

//	вычисление нормы вектора
double normV(double *v, int size)
{
	double norm = 0.;
#pragma omp parallel for
	for (int i = 0; i < size; i++)
		norm += v[i] * v[i];
	return norm;
}

//	умножение матрицы на вектор
double* mv_mult(rrco m, double *v, int size)
{
	double *result = new double[size];
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		double val = 0.;
		for (int j = 0; j < m.matrix[i].size(); j++)
		{
			val += m.matrix[i].at(j).value * v[m.matrix[i].at(j).column];
		}
		result[i] = val;
	}
	return result;
}

//	решение симметричной положительно определенной системы уравнений методом сопряженных градиентов
double* conjurate_gradient_method(rrco m, double *f, int n)
{
	//	выделение памяти
	double *zk	= new double[n];
	double *rk	= new double[n];
	double *sz	= new double[n];
	double *x	= new double[n];

	double alpha, beta, mf;
	double sp, sp1, spz;

	int i = 0, j = 0, kl = 1;
	//	вычисление нормы вектора - правой части
	mf = normV(f, n);
	//	установим начальное приближение
	for (i = 0; i < n; i++)
		x[i] = 0.2;
	//	зададим начальные значения r0, z0
	sz = mv_mult(m, x, n);
#pragma omp parallel for
	for (i = 0; i < n; i++)
	{
		rk[i] = f[i] - sz[i];
		zk[i] = rk[i];
	}
	int iter = 0;
	//	последовательность итераций	
	do
	{
		iter++;
		//	вычисление числителя и знаменателя для коэффициента
		spz = 0;
		sp = 0;
		sz = mv_mult(m, zk, n);
#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			spz += sz[i] * zk[i];
			sp += rk[i] * rk[i];
		}
		alpha = sp / spz;
		/* Вычисление вектора решения xk = xk-1 + alpha * zk - 1,
		* вектора невязки rk = rk-1 - alpha * A * zk - 1
		* и числителя для beta, равного норме вектора невязки
		*/
		sp1 = 0;
#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			x[i] += alpha * zk[i];
			rk[i] -= alpha * sz[i];
			sp1 += rk[i] * rk[i];
		}
		kl++;
		//	вычисление beta
		beta = sp1 / sp;
		// вычисление вектора спуска
#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			zk[i] = rk[i] + beta * zk[i];
		}

	} while (sp1 / mf > EPS * EPS / C / C && iter < MAX_ITER);
	printf("Conjurate gradient method - iterations = %d\n", iter);
	delete[] zk;
	delete[] rk;
	delete[] sz;

	return x;
}

//	метод сопряженных градиентов с диагональным предобусловливанием
double *CGM_precond(rrco m, double *f, int n)
{
	//	выделение памяти
	double *zk = new double[n];
	double *rk = new double[n];
	double *sz = new double[n];
	double *drk = new double[n];
	double *x = new double[n];

	double alpha, beta, mf;
	double sp, sp1, spz;

	int i = 0, j = 0, kl = 1;
	//	вычисление нормы вектора - правой части
	mf = normV(f, n);
	//	установим начальное приближение
	for (i = 0; i < n; i++)
		x[i] = 0.2;
	//	зададим начальные значения r0, z0
	sz = mv_mult(m, x, n);
#pragma omp parallel for
	for (i = 0; i < n; i++)
	{
		rk[i] = f[i] - sz[i];
		zk[i] = rk[i] / m.at(i, i);
	}
	int iter = 0;
	//	последовательность итераций	
	do
	{
		iter++;
		//	вычисление числителя и знаменателя для коэффициента
		spz = 0;
		sp = 0;
		sz = mv_mult(m, zk, n);
#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			drk[i] = rk[i] / m.at(i, i);
			spz += sz[i] * zk[i];
			sp += drk[i] * rk[i];
		}
		alpha = sp / spz;
		/* Вычисление вектора решения xk = xk-1 + alpha * zk - 1,
		* вектора невязки rk = rk-1 - alpha * A * zk - 1
		* и числителя для beta, равного норме вектора невязки
		*/
		sp1 = 0;
#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			x[i] += alpha * zk[i];
			rk[i] -= alpha * sz[i];
		}
#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			drk[i] = rk[i] / m.at(i, i);
			sp1 += drk[i] * rk[i];
		}
		kl++;
		//	вычисление beta
		beta = sp1 / sp;
		// вычисление вектора спуска
#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			zk[i] = drk[i] + beta * zk[i];
		}

	} while (sp1 / mf > EPS * EPS / C / C && iter < MAX_ITER);
	printf("Conjurate gradient method - iterations = %d\n", iter);
	delete[] zk;
	delete[] rk;
	delete[] sz;
	delete[] drk;

	return x;
}

double lamdba(double x, double y)
{
	if (x < 5.)
		return 1.;
	else if (x < 10.)
		return 2.;
	else return 3.;
}

double gamma(double x, double y)
{
	if (x < 5.)
		return 1.;
	else if (x < 10.)
		return 2.;
	else return 3.;
	//return 3.;
}

//	аналитическое решение уравнения
double U(double x, double y)
{
	return x*x + y*y;
}

//	функция, стоящая в правой части уравнения
double F(double x, double y)
{
	return gamma(x, y) * U(x, y) - lamdba(x, y) * 2;
}

int main(int argc, char *argv[])
{
	FILE *input = fopen("input.txt", "r");
	input_data dat = input_data();
	read_input_data(input, dat);
	fclose(input);
	//	определим число узлов сетки по X и Y
	int cx = 0, cy = 0;
	double sx = dat.stepX, sy = dat.stepY;
	double curr = dat.x0;
	while (curr <= dat.xF)
	{
		curr += sx;
		sx *= dat.coeffX;
		cx++;
	}
	curr = dat.y0;
	while (curr <= dat.yF)
	{
		curr += sy;
		sy *= dat.coeffY;
		cy++;
	}
	printf("Nodes by X = %d\n", cx);
	printf("Nodes by Y = %d\n", cy);
	//	выделим память для хранения неравномерной сетки
	int n = cx * cy;	//	количество узлов сетки
	int m = (cx - 1) * (cy - 1);	//	количество конечных элементов
	double **mesh = new double*[n];	//	массив координат узлов сетки
	for (int i = 0; i < n; i++)
		mesh[i] = new double[2];
	//	Сформируем сетку
	sx = 0; sy = 0;
	double tempX = dat.stepX, tempY = dat.stepY;
	for (int i = cy - 1; i >= 0; i--)
	{
		dat.stepX = tempX; sx = 0;
		for (int j = cx - 1; j >= 0; j--)
		{
			if (j == cx - 1)
				mesh[i * cx + j][0] = dat.xF;
			else
				mesh[i * cx + j][0] = dat.xF - sx;
			if (i == cy - 1)
				mesh[i * cx + j][1] = dat.yF;
			else
				mesh[i * cx + j][1] = dat.yF - sy;
			sx += dat.stepX;
			dat.stepX *= dat.coeffX;
		}
		sy += dat.stepY;
		dat.stepY *= dat.coeffY;
	}
	//	Сформируем массив элементов
	int **elements = new int*[m];
	for (int i = 0; i < m; i++)
		elements[i] = new int[4];
	//	вложенный цикл
	int temp = 0;
	for (int j = 0; j < cy - 1; j++)
	{
		for (int i = 0; i < cx - 1; i++)
		{
			elements[temp][0] = j * (cx)+i;
			elements[temp][1] = j * (cx)+i + 1;
			elements[temp][2] = (j + 1) * (cx)+i;
			elements[temp][3] = (j + 1) * (cx)+i + 1;
			temp++;
		}
	}
	//	выделим память для хранения глобальной матрицы системы
	rrco A;
	A = rrco();
	A.memset(n);
	//	выделим память для хранения глобального вектора правой части
	double *f = new double[n];
	double *u = new double[n];
	for (int i = 0; i < n; i++)
		f[i] = u[i] = 0.;
	//	для формирования глобальной матрицы системы пройдем циклом по элементам
	double coeff, coeff1, coeff2;
	double l = 0., g = 0.;
	double stepX, stepY;
	//	локальная матрица жесткости
	double G[4][4];
	//	локальная матрица масс
	double M[4][4];
	//	локальный вектор правой части
	double b[4];
	for (int i = 0; i < m; i++)
	{
		stepX = mesh[elements[i][0]][0] - mesh[elements[i][1]][0];
		stepY = mesh[elements[i][0]][1] - mesh[elements[i][2]][1];

		//	вычислим значения lambda, gamma внутри каждого конечного элемента
		//l = lamdba(mesh[elements[i][0]][0] + (stepX / 2), mesh[elements[i][0]][1] + (stepY / 2));
		//g = gamma(mesh[elements[i][0]][0] + (stepX / 2), mesh[elements[i][0]][1] + (stepY / 2));

		l = lamdba(mesh[elements[i][0]][0], mesh[elements[i][0]][1]);
		g = gamma(mesh[elements[i][0]][0], mesh[elements[i][0]][1]);

		//printf("Lambda = %4.3lf, gamma = %4.3lf\n", l, g);

		coeff1 = (l / 6) * (stepY / stepX);
		coeff2 = (l / 6) * (stepX / stepY);
		//	формируем локальную матрицу жесткости
		G[0][0] = coeff1 * 2 + coeff2 * 2;
		G[0][1] = coeff1 * (-2) + coeff2;
		G[0][2] = coeff1 + coeff2 * (-2);
		G[0][3] = -coeff1 - coeff2;
		G[1][0] = -coeff1 * 2 + coeff2;
		G[1][1] = coeff1 * 2 + coeff2 * 2;
		G[1][2] = -coeff1 - coeff2;
		G[1][3] = coeff1 - coeff2 * 2;
		G[2][0] = coeff1 - coeff2 * 2;
		G[2][1] = -coeff1 - coeff2;
		G[2][2] = coeff1 * 2 + coeff2 * 2;
		G[2][3] = -coeff1 * 2 + coeff2;
		G[3][0] = -coeff1 - coeff2;
		G[3][1] = coeff1 - coeff2 * 2;
		G[3][2] = -coeff1 * 2 + coeff2;
		G[3][3] = coeff1 * 2 + coeff2 * 2;
		//	формируем локальную матрицу масс
		coeff = g * stepX * stepY / 36;
		M[0][0] = coeff * 4;	M[0][1] = coeff * 2;	M[0][2] = coeff * 2;	M[0][3] = coeff * 1;
		M[1][0] = coeff * 2;	M[1][1] = coeff * 4;	M[1][2] = coeff * 1;	M[1][3] = coeff * 2;
		M[2][0] = coeff * 2;	M[2][1] = coeff * 1;	M[2][2] = coeff * 4;	M[2][3] = coeff * 2;
		M[3][0] = coeff * 1;	M[3][1] = coeff * 2;	M[3][2] = coeff * 2;	M[3][3] = coeff * 4;

		//	формируем локальный вектор правой части
		//	пересчитаем коэффициент по формуле
		coeff = stepX * stepY / 36;
		//	определим значения F_i в узлах элемента
		double f0 = F(mesh[elements[i][0]][0], mesh[elements[i][0]][1]);
		double f1 = F(mesh[elements[i][1]][0], mesh[elements[i][1]][1]);
		double f2 = F(mesh[elements[i][2]][0], mesh[elements[i][2]][1]);
		double f3 = F(mesh[elements[i][3]][0], mesh[elements[i][3]][1]);
		//	считаем локальный вектор правой части
		b[0] = coeff * (f0 * 4 + f1 * 2 + f2 * 2 + f3);
		b[1] = coeff * (2 * f0 + 4 * f1 + f2 + 2 * f3);
		b[2] = coeff * (2 * f0 + f1 + 4 * f2 + 2 * f3);
		b[3] = coeff * (f0 + 2 * f1 + 2 * f2 + 4 * f3);
		//	необходимо сформировать глобальную матрицу системы
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				A.add(G[j][k] + M[j][k], elements[i][j], elements[i][k]);
			}
		}
		//	сформируем глобальный вектор правой части
		f[elements[i][0]] += b[0];
		f[elements[i][1]] += b[1];
		f[elements[i][2]] += b[2];
		f[elements[i][3]] += b[3];
	}
	FILE *out = fopen("matrix.txt", "w");
	A.print_to_file(out);
	fclose(out);
	
	//	Учтем граничные условия первого рода
	int fact_n = n;
	/*for (int i = 0; i < n; i++)
	{
		if (mesh[i][0] == dat.x0)
		{
			//A.matrix[i].clear();
			A.add(C - A.at(i, i), i, i);
			f[i] = U(mesh[i][0], mesh[i][1]) * C;
			/*for (int k = 0; k < A.matrix.size(); k++)
			{
				if (A.at(k, i) != NULL)
				{
					double v = A.at(k, i);
					double temp = A.at(k, i) * U(mesh[i][0], mesh[i][1]);
					f[k] -= temp;
					for (int r = 0; r < A.matrix[k].size(); r++)
					{
						if (A.matrix[k].at(r).value == v)
							A.matrix[k].erase(A.matrix[k].begin() + r);
					}
				}
			}
			A.matrix.erase(A.matrix.begin() + i);
			double *temp = new double[n - 1];
			for (int k = 0; k < i; k++)
				temp[i] = f[i];
			for (int k = i + 1; k < n; k++)
				temp[i] = f[i];
			f = temp;
			delete[] temp;
			u[i] = U(mesh[i][0], mesh[i][1]);
			n--; // здесь был конец комментария
		}
		if (mesh[i][0] == dat.xF)
		{
			//A.matrix[i].clear();
			A.add(C - A.at(i, i), i, i);
			f[i] = U(mesh[i][0], mesh[i][1]) * C;
		}
		if (mesh[i][1] == dat.y0)
		{
			//A.matrix[i].clear();
			A.add(C - A.at(i, i), i, i);
			f[i] = U(mesh[i][0], mesh[i][1]) * C;
		}
		if (mesh[i][1] == dat.yF)
		{
			//A.matrix[i].clear();
			A.add(C - A.at(i, i), i, i);
			f[i] = U(mesh[i][0], mesh[i][1]) * C;
		}
	}*/
	//	решение дискретного аналога методом сопряженных градиентов
	//u = conjurate_gradient_method(A, f, n);
	u = CGM_precond(A, f, n);
	int j = 0;
	double e1 = 0., e2 = 0.;
	FILE *output = fopen("output.txt", "w");
	for (int i = 0; i < n; i++)
	{
		fprintf(output, "%4.12lf\t%4.12lf\t%4.12lf\t%4.3lf\t%4.3lf\n", u[i], U(mesh[i][0], mesh[i][1]), abs(u[i] - U(mesh[i][0], mesh[i][1])), mesh[i][0], mesh[i][1]);
		e1 += abs(u[i] - U(mesh[i][0], mesh[i][1]));
		e2 += abs(U(mesh[i][0], mesh[i][1]));
	}
	fprintf(output, "\nНорма относительной погрешности\t= %2.18lf\n", sqrt(e1*e1/e2/e2));
	fclose(output);
	//	освободим память
	for (int i = 0; i < n; i++)
		delete[] mesh[i];
	delete[] mesh;
	for (int i = 0; i < m; i++)
		delete[] elements[i];
	delete[] elements;
	A.clear();
	delete[] f;
	delete[] u;
	system("pause");
	return EXIT_SUCCESS;
}