/*
	Решение стационарной задачи теплопроводности в однородной трехмерной расчетной области
	методом конечных элементов
	-	восьмиугольные конечные элементы
	-	трилинейные базисные функции на элементе
	-	равномерная сетка
	-	разреженный строчный формат хранения матриц
	-	решатель - метод сопряженных градиентов с предобусловливанием
	Захаров С.А., 14-09-2017
*/

#define _CRT_SECURE_NO_WARNINGS
#ifndef ulong
#define ulong unsigned long
#endif // !ulong
#ifndef MAX_ITER
#define MAX_ITER 20000
#endif // !MAX_ITER

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>

using namespace std;

const double EPS = 1e-16;

const double G1[2][2] = { {1, -1}, {-1, 1} };
const double M1[2][2] = { {1. / 3., 1. / 6.}, {1. / 6., 1. / 3.} };

struct mat_el
{
	double value;
	ulong column;

	mat_el(double v, ulong c)
	{
		value = v;
		column = c;
	}
};

//	разреженный строчный формат представления матрицы
struct rrco
{
	//	количество уравнений в системе
	ulong N;
	//	элементы матрицы по строкам
	vector<vector<mat_el>> matrix;

	rrco()
	{
		N = 0;
		matrix.clear();
	}

	//	выделение памяти
	void memset(ulong n)
	{
		N = n;
		matrix.resize(n);
		for (ulong i = 0; i < n; i++)
			matrix[i].clear();
	}

	//	очистка памяти
	void clear()
	{
		for (ulong i = 0; i < N; i++)
		{
			matrix[i].clear();
		}
		matrix.clear();
	}

	double at(ulong row, ulong column)
	{
		for (ulong i = 0; i < matrix[row].size(); i++)
			if (column == matrix[row].at(i).column)
				return matrix[row].at(i).value;
		return NULL;
	}

	void add(double value, ulong row, ulong column)
	{
		for (ulong i = 0; i < matrix[row].size(); i++)
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
		for (ulong i = 0; i < N; i++)
		{
			for (ulong j = 0; j < N; j++)
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
		for (ulong i = 0; i < N; i++)
		{
			for (ulong j = 0; j < N; j++)
			{
				v = at(i, j);
				if (v != 0.)
					fprintf(output, "%4.3lf\t", v);
				else
					fprintf(output, "0\t");
			}
			fprintf(output, "\n");
		}
	}
};

struct input_data
{
	double x0, y0, z0, xF, yF, zF;
	double stepX, stepY, stepZ;

	input_data()
	{
		x0 = y0 = z0 = xF = yF = zF = 0.;
		stepX = stepY = stepZ = 0.;
	}

	void read(FILE *in)
	{
		if (in == NULL)
		{
			printf("Cannot read input data\n");
			return;
		}
		fscanf(in, "%lf %lf %lf", &x0, &y0, &z0);
		fscanf(in, "%lf %lf %lf", &xF, &yF, &zF);
		fscanf(in, "%lf %lf %lf", &stepX, &stepY, &stepZ);
	}
};

//	вычисление нормы вектора
double normV(double *v, ulong size)
{
	double norm = 0.;
	for (ulong i = 0; i < size; i++)
		norm += v[i] * v[i];
	return norm;
}

//	умножение матрицы на вектор
double* mv_mult(rrco m, double *v, ulong size)
{
	double *result = new double[size];
	for (ulong i = 0; i < size; i++)
	{
		double val = 0.;
		for (ulong j = 0; j < m.matrix[i].size(); j++)
		{
			val += m.matrix[i].at(j).value * v[m.matrix[i].at(j).column];
		}
		result[i] = val;
	}
	return result;
}

//	метод сопряженных градиентов с диагональным предобусловливанием
double *CGM_precond(rrco m, double *f, ulong n)
{
	//	выделение памяти
	double *zk = new double[n];
	double *rk = new double[n];
	double *sz = new double[n];
	double *drk = new double[n];
	double *x = new double[n];

	double alpha, beta, mf;
	double sp, sp1, spz;

	ulong i = 0, j = 0, kl = 1;
	//	вычисление нормы вектора - правой части
	mf = normV(f, n);
	//	установим начальное приближение
	for (i = 0; i < n; i++)
		x[i] = 0.2;
	//	зададим начальные значения r0, z0
	sz = mv_mult(m, x, n);
	for (i = 0; i < n; i++)
	{
		rk[i] = f[i] - sz[i];
		zk[i] = rk[i] / m.at(i, i);
	}
	ulong iter = 0;
	//	последовательность итераций	
	do
	{
		iter++;
		//	вычисление числителя и знаменателя для коэффициента
		spz = 0;
		sp = 0;
		sz = mv_mult(m, zk, n);
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
		for (i = 0; i < n; i++)
		{
			x[i] += alpha * zk[i];
			rk[i] -= alpha * sz[i];
		}
		for (i = 0; i < n; i++)
		{
			drk[i] = rk[i] / m.at(i, i);
			sp1 += drk[i] * rk[i];
		}
		kl++;
		//	вычисление beta
		beta = sp1 / sp;
		// вычисление вектора спуска
		for (i = 0; i < n; i++)
		{
			zk[i] = drk[i] + beta * zk[i];
		}

	} while (sp1 / mf > EPS * EPS && iter < MAX_ITER);
	printf("Conjurate gradient method - iterations = %d\n", iter);
	delete[] zk;
	delete[] rk;
	delete[] sz;
	delete[] drk;

	return x;
}

double lambda(double x, double y, double z)
{
	return 2.;
}

double gamma(double x, double y, double z)
{
	return 3.;
}

//	аналитическое решение
double U(double x, double y, double z)
{
	return 3. * x + 4. * y - 2. * z;
}

//	функция, стоящая в правой части уравнения
double F(double x, double y, double z)
{
	return gamma(x, y, z) * U(x, y, z);
}

//	метод Гаусса для тестирования решения
double *gauss(double **A, double *f, ulong n)
{
	double r = 0., *res;
	res = new double[n];
	//	прямой ход
	for (int i = 0; i < n; i++)
	{
		for (int j = i + 1; j < n; j++)
		{
			A[j][i] = -A[j][i] / A[i][i];
			for (int k = i + 1; k < n; k++)
			{
				A[j][k] = A[j][k] + A[j][i] * A[i][k];
			}
			f[j] += A[j][i] * f[i];
		}
	}
	//	обратный ход
	res[n - 1] = f[n - 1] / A[n - 1][n - 1];
	for (int i = n - 2; i >= 0; i--)
	{
		r = f[i];
		for (int j = i + 1; j < n; j++)
		{
			r -= res[j] * A[i][j];
		}
		res[i] = r / A[i][i];
	}
	return res;
}

//	функции для соответствия локальной и глобальной нумерации

int _m(int i)
{
	return (i % 2);
}

int _v(int i)
{
	return (i / 2) % 2;
}

int _w(int i)
{
	return i / 4;
}

int main(int argc, char *argv[])
{
	//	считаем входные данные
	FILE *input = fopen("input3d.txt", "r");
	input_data dat;
	dat = input_data();
	dat.read(input);
	fclose(input);
	//	количество узлов сетки
	ulong n = 0;
	n += (ulong)(dat.xF - dat.x0) / dat.stepX;
	n++;
	n *= (ulong)((dat.yF - dat.y0) / dat.stepY + 1);
	n *= (ulong)((dat.zF - dat.z0) / dat.stepZ + 1);
	printf("%d\n", n);
	//	массив координат узлов сетки
	double **mesh = new double*[n];
	for (ulong i = 0; i < n; i++)
		mesh[i] = new double[3];

	//	локальная матрица жесткости
	double G[8][8];

	//	локальная матрица масс
	double M[8][8];

	//	глобальная матрица системы (обычный формат)
	double **A = new double*[n];
	for (ulong i = 0; i < n; i++)
	{
		A[i] = new double[n];
		for (ulong j = 0; j < n; j++)
			A[i][j] = 0.;
	}

	//	генерация равномерной сетки
	ulong temp = 0;
	for (ulong i = 0; i < (ulong)((dat.zF - dat.z0) / dat.stepZ + 1); i++)
	{
		for (ulong j = 0; j < (ulong)((dat.yF - dat.y0) / dat.stepY + 1); j++)
		{
			for (ulong k = 0; k < (ulong)((dat.xF - dat.x0) / dat.stepX + 1); k++)
			{
				mesh[temp][0] = dat.x0 + dat.stepX * k;
				mesh[temp][1] = dat.y0 + dat.stepY * j;
				mesh[temp][2] = dat.z0 + dat.stepZ * i;
				//	вывод сетки в консоль
				printf("%d - %4.3lf\t%4.3lf\t%4.3lf\n", temp, mesh[temp][0], mesh[temp][1], mesh[temp][2]);
				temp++;
			}
		}
	}
	//	количество конечных элементов
	ulong m = (ulong)((dat.zF - dat.z0) / dat.stepZ);
	m *= (ulong)((dat.yF - dat.y0) / dat.stepY);
	m *= (ulong)((dat.xF - dat.x0) / dat.stepX);
	printf("Number of elements = %d\n", m);
	//	массив конечных элементов
	ulong **elements = new ulong*[m];
	for (ulong i = 0; i < m; i++)
		elements[i] = new ulong[8];
	//	количество узлов сетки по измерениям
	ulong cz = (ulong)((dat.zF - dat.z0) / dat.stepZ + 1);
	ulong cy = (ulong)((dat.yF - dat.y0) / dat.stepY + 1);
	ulong cx = (ulong)((dat.xF - dat.x0) / dat.stepX + 1);
	//	локальная нумерация узлов в конечных элементах
	temp = 0;
	for (ulong i = 0; i < cz - 1; i++)
	{
		for (ulong j = 0; j < cy - 1; j++)
		{
			for (ulong k = 0; k < cx - 1; k++)
			{
				elements[temp][0] = i * cy * cx + j * cx + k;
				elements[temp][1] = i * cy * cx + j * cx + k + 1;
				elements[temp][2] = i * cy * cx + (j + 1) * cx + k;
				elements[temp][3] = i * cy * cx + (j + 1) * cx + k + 1;
				elements[temp][4] = (i + 1) * cy * cx + j * cx + k;
				elements[temp][5] = (i + 1) * cy * cx + j * cx + k + 1;
				elements[temp][6] = (i + 1) * cy * cx + (j + 1) * cx + k;
				elements[temp][7] = (i + 1) * cy * cx + (j + 1) * cx + k + 1;
				temp++;
			}
		}
	}

	//	цикл по элементам
	for (ulong t = 0; t < m; t++)
	{
		double stepX, stepY, stepZ;
		stepX = abs(mesh[elements[t][0]][0] - mesh[elements[t][1]][0]);
		stepY = abs(mesh[elements[t][0]][1] - mesh[elements[t][2]][1]);
		stepZ = abs(mesh[elements[t][0]][2] - mesh[elements[t][6]][2]);
		//	TODO - формирование локальной матрицы жесткости и масс
		printf("\n");
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				G[i][j] = lambda(0, 0, 0) * G1[_m(i)][_m(j)] * M1[_v(i)][_v(j)] * M1[_w(i)][_w(j)] * stepY * stepZ / stepX;
				G[i][j] += lambda(0, 0, 0) * M1[_m(i)][_m(j)] * G1[_v(i)][_v(j)] * M1[_w(i)][_w(j)] * stepX * stepZ / stepY;
				G[i][j] += lambda(0, 0, 0) * M1[_m(i)][_m(j)] * M1[_v(i)][_v(j)] * G1[_w(i)][_w(j)] * stepY * stepX / stepZ;
				//printf("%4.3lf\t", G[i][j]);

				M[i][j] = gamma(0, 0, 0) * M1[_m(i)][_m(j)] * M1[_v(i)][_v(j)] * M1[_w(i)][_w(j)] * stepX * stepY * stepZ;
				printf("%4.3lf\t", M[i][j]);
			}
			printf("\n");
		}
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				A[elements[t][j]][elements[t][k]] += G[j][k] + M[j][k];
			}
		}
	}

	FILE *test = fopen("test.txt", "w");
	for (ulong i = 0; i < n; i++)
	{
		for (ulong j = 0; j < n; j++)
			//fprintf(test, "%4.3lf\t", A[i][j]);
			if (A[i][j]) fprintf(test, "*"); else fprintf(test, " ");
		fprintf(test, "\n");
	}
	fclose(test);
	
	//	освободим память
	for (ulong i = 0; i < m; i++)
		delete[] elements[i];
	delete[] elements;
	for (ulong i = 0; i < n; i++)
		delete[] mesh[i];
	delete[] mesh;

	system("pause");
	return EXIT_SUCCESS;
}