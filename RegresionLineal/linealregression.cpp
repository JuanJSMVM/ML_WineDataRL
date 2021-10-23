#include "linealregression.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/* Se necesita entrenar el modelo, lo que implica  minimizar
 * alguna función de costo y de esta forma se puede medir la
 * precisión de la función de hipotesis. La función de costo es la forma
 * de penalizar al modelo por cometer un error
 */


float LinealRegression::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia = pow((X* theta-y).array(),2);
    return diferencia.sum()/(2*X.rows());
}

/* Se implementa la función para dar al algoritmo los valores de theta iniciales, que
 * cambiaran iterativamente hasta que converga al valor minimo de la función de costo.
 * Basicamente describira el gradiente descendiente: el cual es dado por la derivada
 * parcial de la función.
 * La función tiene un alpha que representa el salto del gradiente y el número de iteraciones
 * que se necesitan para actualizar theta  hasta que la función converja al minimo esperado.
 */

std::tuple<Eigen::VectorXd, std::vector<float>> LinealRegression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta,float alpha, int iteraciones){
    // Almacenamiento temporal para los valores de theta.
    Eigen::MatrixXd temporal = theta;

    //Variable con la cantidad de parametros m (FEATURES).
    int parametros = theta.rows();

    //Ubicar el costo inicial, que se actualizara iterativamente con los pesos
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X,y,theta));

    //Por cada iteración se calcula la función de error
    for (int i = 0; i < iteraciones; ++i) {
        Eigen::MatrixXd error = X*theta-y;
        for (int j = 0; j < parametros; ++j) {
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = theta(j,0)-(alpha/X.rows())*termino.sum();
        }
        theta = temporal;
        costo.push_back(FuncionCosto(X,y,theta));
    }
    return std::make_tuple(theta,costo);
}
