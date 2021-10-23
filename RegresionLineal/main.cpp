#include "exeigennorm.h"
#include "linealregression.h"
#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

/* En primer lugar, se creara una clase llamada "ExEigenNorm", la cual nos permitira
 * leer un dataset, extraer los datos, montar sobre la estructura Eigen para normalizar los datos
 *
*/

int main(int argc, char *argv[])
{
    /* Se crea un objeto del tipo ExEigenNorm, y se incluyen los tres elementos del constructor:
       nombre del dataset, delimitador, flag(Si tiene o no tiene header)*/

    ExEigenNorm extraccion(argv[1],argv[2],argv[3]);
    LinealRegression LR;



    /* Se leen los datos del archivo por la función LeerCSV*/
    std::vector<std::vector<std::string>> dataFrame = extraccion.LeerCSV();

    /*
     * Para probar la segunda función CSVtoEigen se define la cantidad de filas y columnas
     * basados en los datos de entrada.
     */
    int filas =dataFrame.size();
    int columnas = dataFrame[0].size();
    Eigen::MatrixXd matrizDataF = extraccion.CSVtoEigen(dataFrame,filas,columnas);

    //std::cout<<"////////////////////////////////////////////////////////////////"<<std::endl;
    //std::cout<<matrizDataF<<std::endl;
    //std::cout<<"////////////////////////////////////////////////////////////////"<<std::endl;

    /*
     * Para desarrollar el primer algoritmo de regresion lineal, en donde se probara con los datos
     * de los vinos(winedata.csv) se presenta la regresion lineal para multiples variables.
     * Dada la naturaleza de la regresion lineal, si se tiene variables con diferentes unidades,
     * una variable podria beneficiar/estropear otra variable: se necesitará estandarizar los datos,
     * dejando a todas las variables del mismo orden de magnitud y centradas en cero. Para ello se
     * construirá una función de normalización basada en el setscore normalización. Se necesita tres
     * funciones: la función de normalización, la del promedio y la desviación estandar.
     */

    Eigen::MatrixXd normalizados = extraccion.Normalizacion(matrizDataF);
    //std::cout<<"Normalizados: "<<std::endl<<std::endl;

    /*
     * Se imprimen las 10 primeras filas normalizadas del dataframe.
     */
    //std::cout<<normalizados;

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> divDatos = extraccion.TrainTestSplit(normalizados,0.8);
    /*
     * Se desempaca la tupla, se usa std::tie
     * https://en.cppreference.com/w/cpp/utility/tuple/tie
     */
    Eigen::MatrixXd X_train,y_train, X_test, y_test;


    std::tie(X_train,y_train, X_test, y_test) = divDatos;

    /*
     * Inspeccion visual de la división de los datos para entrenamiento y prueba.
     */

    std::cout<<"\n\nTamaño original->                   "<<normalizados.rows()<<std::endl;
    std::cout<<"Tamaños variables dependientes:     "<<std::endl;
    std::cout<<"Tamaño Entrenamiento X(filas)->     "<<X_train.rows()<<std::endl;
    std::cout<<"Tamaño Entrenamiento X(cols)->      "<<X_train.cols()<<std::endl;
    std::cout<<"Tamaño Prueba (filas)->             "<<X_test.rows()<<std::endl;
    std::cout<<"Tamaño Prueba (cols)->              "<<X_test.cols()<<std::endl;

    std::cout<<"\nTamaños variable independiente:   "<<std::endl;
    std::cout<<"Tamaño Entrenamiento X(filas)->     "<<y_train.rows()<<std::endl;
    std::cout<<"Tamaño Entrenamiento X(cols)->      "<<y_train.cols()<<std::endl;
    std::cout<<"Tamaño Prueba (filas)->             "<<y_test.rows()<<std::endl;
    std::cout<<"Tamaño Prueba (cols)->              "<<y_test.cols()<<std::endl;

    // A continuación se procede a crear la clase LinealRegression.
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /* Redimensión de las matrices para ubicación en los vectores de ONES (similar a reshape con numpy) */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /* Se define el vector theta que pasara al algoritmo de gradiente descendiente (basicamente un vector
     * de ZEROS del mismo tamaño del vector de entrenamiento. Adicional se pasara alpha y el número de iteraciones
     */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    // Se definen las variables de salida que representan los coeficientes y el vector de costo
    Eigen::VectorXd thetaOut;
    std::vector<float> costo;
    std::tuple<Eigen::VectorXd,std::vector<float>> gradienteD = LR.GradienteDescendiente(X_train,y_train,theta,alpha,iteraciones);
    std::tie(thetaOut,costo) = gradienteD;

    // Se imprimen los valores de los coeficientes theta para cada FEATURE.
    std::cout<<"\nTheta: \n"<<thetaOut<<std::endl;

    std::cout<<"\nCosto: \n"<<std::endl;
    for (auto valor : costo) {
        std::cout<<valor<<std::endl;
    }

    /*
     * Exportamos a ficheros, costo y thetaOut
     */
    //extraccion.VectorToFile(costo,"Costo.txt");
    //extraccion.EigenToFile(thetaOut,"ThetaOut.txt");
    return EXIT_SUCCESS;
}
