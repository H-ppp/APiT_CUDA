#include <windows.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <immintrin.h> // Для SSE и AVX
#include <omp.h> // Для OpenMP

using namespace std;

// Функция для получения размеров кэша L1, L2 и L3
void GetCacheSizes(size_t& l1CacheSize, size_t& l2CacheSize, size_t& l3CacheSize) {
    DWORD bufferSize = 0;
    GetLogicalProcessorInformation(nullptr, &bufferSize);

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
        cerr << "Ошибка при получении информации о процессоре." << std::endl;
        return;
    }

    for (const auto& info : buffer) {
        if (info.Relationship == RelationCache) {
            switch (info.Cache.Level) {
            case 1:
                l1CacheSize += info.Cache.Size;
                break;
            case 2:
                l2CacheSize += info.Cache.Size;
                break;
            case 3:
                l3CacheSize += info.Cache.Size;
                break;
            default:
                break;
            }
        }
    }
}

// Функция для умножения матриц стандартным методом (Реализация 1)
void standardMatrixMultiply(float** A, float** B, float** C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Функция для умножения матриц блочным методом (Реализация 2,3)
void blockMatrixMultiply(float** A, float** B, float** C, int N, int blockSize) {
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, N); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, N); ++jj) {
                        float sum = 0;
                        for (int kk = k; kk < min(k + blockSize, N); ++kk) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] += sum;
                    }
                }
            }
        }
    }
}

// Векторизированная функция для умножения матриц (Реализация 4)
void blockVectorizedMatrixMultiply(float** A, float** B, float** C, int N, int blockSize) {
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, N); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, N); jj += 4) {
                        __m128 c = _mm_loadu_ps(&C[ii][jj]);
                        for (int kk = k; kk < min(k + blockSize, N); ++kk) {
                            __m128 a = _mm_set1_ps(A[ii][kk]);
                            __m128 b = _mm_loadu_ps(&B[kk][jj]);
                            c = _mm_add_ps(c, _mm_mul_ps(a, b));
                        }
                        _mm_storeu_ps(&C[ii][jj], c);
                    }
                }
            }
        }
    }
}
// OpenMP векторизированная функция для умножения матриц (Реализация 5)
void ompBlockVectorizedMatrixMultiply(float** A, float** B, float** C, int N, int blockSize) {
#pragma omp parallel for
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, N); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, N); jj += 4) {
                        __m128 c = _mm_loadu_ps(&C[ii][jj]); // Загружаем значения из C
                        for (int kk = k; kk < min(k + blockSize, N); ++kk) {
                            __m128 a = _mm_set1_ps(A[ii][kk]); // Загружаем элемент из A
                            __m128 b = _mm_loadu_ps(&B[kk][jj]); // Загружаем блок из B
                            c = _mm_add_ps(c, _mm_mul_ps(a, b)); // Умножение и сложение
                        }
                        _mm_storeu_ps(&C[ii][jj], c); // Сохраняем результат в C
                    }
                }
            }
        }
    }
}


// Функция для сравнения результатов двух матриц
bool compareMatrices(float** A, float** B, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(A[i][j] - B[i][j]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

// Функция для создания матрицы
float** createMatrix(int N) {
    float** matrix = new float* [N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new float[N];
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = static_cast<float>(rand() % 100);
        }
    }
    return matrix;
}

// Функция для обнуления матрицы
void zeroMatrix(float** matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = 0.0f;
        }
    }
}

// Функция для удаления матрицы
void deleteMatrix(float** matrix, int N) {
    for (int i = 0; i < N; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

int calculateBlockSize(size_t cacheSize) {
    // Шаг 1: Вычисляем максимальный размер блока
    double maxBlockSize = (cacheSize / 3.0) * 0.9;

    // Шаг 2: Делим на sizeof(float)
    maxBlockSize /= sizeof(float);

    // Шаг 3: Находим квадратный корень
    maxBlockSize = sqrt(maxBlockSize);

    // Приводим размер к целому числу
    int blockSize = static_cast<int>(maxBlockSize);

    // Шаг 4: Проверяем кратность 64 и корректируем размер блока
    if (blockSize % 64 != 0) {
        blockSize -= blockSize % 64;
    }

    return blockSize;
}

int main() {
    // Получаем размеры кэшей
    size_t l1CacheSize = 0, l2CacheSize = 0, l3CacheSize = 0;
    GetCacheSizes(l1CacheSize, l2CacheSize, l3CacheSize);
    cout << "L1 Cache Size: " << l1CacheSize / 1024 << " KB" << endl;
    cout << "L2 Cache Size: " << l2CacheSize / 1024 << " KB" << endl;
    cout << "L3 Cache Size: " << l3CacheSize / 1024 << " KB" << endl;

    // Устанавливаем размер матриц исходя из размера L3 кэша
    int N = sqrt(l3CacheSize / sizeof(float)) * 2;
    cout << "Matrix size: " << N << "x" << N << endl;

    // Создаем матрицы
    float** A = createMatrix(N);
    float** B = createMatrix(N);
    float** C1 = createMatrix(N); // Для Реализации 1
    float** C2 = createMatrix(N); // Для Реализации 2
    float** C3 = createMatrix(N); // Для Реализации 3
    float** C4 = createMatrix(N); // Для Реализации 4
    float** C5 = createMatrix(N); // Для Реализации 5

    // Реализация 1: Обычное умножение матриц
    auto start1 = chrono::high_resolution_clock::now();
    standardMatrixMultiply(A, B, C1, N);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> diff1 = end1 - start1;
    cout << "Time for standard multiplication: " << diff1.count() << " s" << endl;
    // Реализация 2: Блочное умножение матриц (L3 кэш)
    zeroMatrix(C2, N);
    int blockSizeL3 = calculateBlockSize(l3CacheSize);
    cout << "Block Size L3: " << blockSizeL3 << "x" << blockSizeL3 << endl;
    auto start2 = chrono::high_resolution_clock::now();
    blockMatrixMultiply(A, B, C2, N, blockSizeL3);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> diff2 = end2 - start2;
    cout << "Time for block multiplication (L3 cache): " << diff2.count() << " s" << endl;

    // Реализация 3: Блочное умножение матриц (L2 кэш)
    zeroMatrix(C3, N);
    int blockSizeL2 = calculateBlockSize(l2CacheSize);
    cout << "Block Size L2: " << blockSizeL2 << "x" << blockSizeL2 << endl;
    auto start3 = chrono::high_resolution_clock::now();
    blockMatrixMultiply(A, B, C3, N, blockSizeL2);
    auto end3 = chrono::high_resolution_clock::now();
    chrono::duration<double> diff3 = end3 - start3;
    cout << "Time for block multiplication (L2 cache): " << diff3.count() << " s" << endl;

    // Реализация 4: Векторизированное умножение
    zeroMatrix(C4, N);
    auto start4 = chrono::high_resolution_clock::now();
    blockVectorizedMatrixMultiply(A, B, C4, N, blockSizeL2);
    auto end4 = chrono::high_resolution_clock::now();
    chrono::duration<double> diff4 = end4 - start4;
    cout << "Time for vectorized block multiplication: " << diff4.count() << " s" << endl;

    // Реализация 5: OpenMP векторизированное умножение
    zeroMatrix(C5, N);
    auto start5 = chrono::high_resolution_clock::now();
    ompBlockVectorizedMatrixMultiply(A, B, C5, N, blockSizeL2);
    auto end5 = chrono::high_resolution_clock::now();
    chrono::duration<double> diff5 = end5 - start5;
    cout << "Time for OpenMP vectorized multiplication: " << diff5.count() << " s" << endl;

    // Проверка равенства матриц
    if (compareMatrices(C1, C2, N)) {
        cout << "Matrices are equal for Implementation 1 and 2!" << endl;
    }
    else {
        cout << "Matrices are NOT equal for Implementation 1 and 2!" << endl;
    }

    if (compareMatrices(C1, C3, N)) {
        cout << "Matrices are equal for Implementation 1 and 3!" << endl;
    }
    else {
        cout << "Matrices are NOT equal for Implementation 1 and 3!" << endl;
    }

    if (compareMatrices(C2, C3, N)) {
        cout << "Matrices are equal for Implementation 2 and 3!" << endl;
    }
    else {
        cout << "Matrices are NOT equal for Implementation 2 and 3!" << endl;
    }

    if (compareMatrices(C1, C4, N)) {
        cout << "Matrices are equal for Implementation 1 and 4!" << endl;
    }
    else {
        cout << "Matrices are NOT equal for Implementation 1 and 4!" << endl;
    }

    if (compareMatrices(C1, C5, N)) {
        cout << "Matrices are equal for Implementation 1 and 5!" << endl;
    }
    else {
        cout << "Matrices are NOT equal for Implementation 1 and 5!" << endl;
    }

    // Освобождаем память
    deleteMatrix(A, N);
    deleteMatrix(B, N);
    deleteMatrix(C1, N);
    deleteMatrix(C2, N);
    deleteMatrix(C3, N);
    deleteMatrix(C4, N);
    deleteMatrix(C5, N);

    system("pause");

    return 0;
}