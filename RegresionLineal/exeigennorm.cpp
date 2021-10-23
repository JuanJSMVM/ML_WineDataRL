#include "exeigennorm.h"

#include <stdlib.h>
#include <vector>
#include <cmath>
#include <boost/algorithm/string.hpp>

/*
Clase usada para saber el tipo de dato de una expresión...
template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}
*/
//https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c/81886


/* Primera función: Lectura de ficheros csv
   vector de vectores string
   La idea es leer linea por linea y almacenar en un vector de vectores tipo String. */
std::vector<std::vector<std::string>> ExEigenNorm::LeerCSV(){
    /* Se abre el archivo para lectura solamente*/
    std::ifstream Archivo(setDatos);
    /* Vector de vectores de tipo String que contendra los datos del dataset. */
    std::vector<std::vector<std::string>> datosString;
    /* Se itera sobre cada linea del dataset y se divide el contenido
       dado por el delimitador provisto por el constructor. */
    std::string linea = "";
    while(getline(Archivo,linea)){
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila,linea,boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    /* Se cierra el archivo. */
    Archivo.close();

    return datosString;
}

/*
 * Se crea la segunda función para guardar el vector de vectores de tipo string
 * a una matrix Eigen. Similar a Pandas(Python) para presentar un dataframe.
*/

Eigen::MatrixXd ExEigenNorm::CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int filas, int col){
    /*
     * Si tiene cabecera la removemos.
     */

    /*
     * Se itera sobre filas y columnas para almacenar en la matrix vacia(Tamaño+filas+columnas),
     * que basicamente almacenará String en un vector: luego lo pasaremos a float para ser manipulados.
     */
    Eigen::MatrixXd dfMatriz(col,filas);
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < col; j++) {
            dfMatriz(j,i) = atof(setDatos[i][j].c_str());
        }
    }
    /*
     * Se transpone la matriz para tener filas por columnas.
     */
    return dfMatriz.transpose();
}

/* A continuación, se van a implementar las funciones para normalización */

/* En c++, la palabra clave auto especifica que el tipo de la variable
 * que se empieza a declarar se deducira automaticamente de su inicializador y,
 * para las funciones si su tipo de retorno es auto, se evaluara mediante la
 * expresión del tipo de retorno en tiempo de ejecución.
 *
 * auto ExEigenNorm::Promedio(Eigen::MatrixXd datos){
 *   Se ingresa como entrada la matriz de datos (datos) y regresa el promedio
 *   return datos.colwise().mean();
 * }
 *
 * Todavia no se sabe que retorna datos.colwise().mean() : En C++, la herencia
 * del tipo de dato no es directa o no se sabe que tipo de dato debe retornar, entonces
 * para ello se declara el tipo en una expresión 'decltype' con el fin de tener
 * seguridad de que tipo de dato retornara la función */

auto ExEigenNorm::Promedio(Eigen::MatrixXd datos)-> decltype (datos.colwise().mean()){
   /* Se ingresa como entrada la matriz de datos (datos) y regresa el promedio */

   return datos.colwise().mean();
}

/* Para implementar la función de desviación estandar
 * datos= x_i - promedio(x) */

auto ExEigenNorm::Desviacion(Eigen::MatrixXd datos)-> decltype(((datos.array().square().colwise().sum())/(datos.rows())).sqrt()){

   //std::cout << "decltype(i) is " << type_name<decltype(((datos.array().square().colwise().sum())/(datos.rows()+1)).sqrt())>() << std::endl;
    return ((datos.array().square().colwise().sum())/(datos.rows())).sqrt();
}

/* Z-score normalizacion es una estrategia de normalizacion de datos para evitar el problema de outlier.
 * Z-score normalization is a strategy of normalizing data that avoids this outlier issue. */
Eigen::MatrixXd ExEigenNorm::Normalizacion(Eigen::MatrixXd datos){

    Eigen::MatrixXd prom = Promedio(datos);

    Eigen::MatrixXd diferenciaPromedio = datos.rowwise()-Promedio(datos);  //(x_i  - promedio)

    std::cout<<"Promedio:"<<std::endl<<std::endl;

    std::cout<<prom<<std::endl<<std::endl;

    //std::cout<<std::endl<<std::endl<<std::endl<<"------------------------------------------"<<std::endl;
    //std::cout<<"Diferencia Promedio:"<<std::endl<<std::endl;
    //std::cout<<diferenciaPromedio<<std::endl<<std::endl;
    /*
     * Variables del tipo auto, usadas para almacenar expresiones de Eigen, no funcionan correctamente.
     * Se recomienda llamar la función directamente o encontrar el tipo de dato de la expresión
     * y declararla explicitamente(Para encontrarla, se puede usar el comando encontrado en la función Desviacion.
     */
    //https://eigen.tuxfamily.org/dox/TopicPitfalls.html#title3
    Eigen::Array<double, 1, -1, 1, 1, -1> desviacion = Desviacion(diferenciaPromedio);
    //std::cout<<"Desviacion:"<<std::endl<<std::endl;
    //std::cout<<desviacion<<std::endl<<std::endl;
    Eigen::MatrixXd matrizNormalizada = diferenciaPromedio.array().rowwise()/(desviacion);


    return matrizNormalizada;
}


/*
 *  A continuación, se hara una función para dividir los datos en conjunto de datos de entrenamiento
 *  y conjunto de datos de prueba.
 */
std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> ExEigenNorm::TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain){
    int filas=datos.rows();
    int filasTrain=round(sizeTrain*filas);
    int filasTest=filas-filasTrain;
    /*
     * Con Eigen, se puede especificar el bloque de una matriz, por ejemplo, se pueden seleccionar
     * las filas superiores para el conjunto de entrenamiento indicando cuantas filas se desean,
     * se selecciona desde 0 (fila 0) hasta el número de filas indicado.
     */
    Eigen::MatrixXd entrenamiento = datos.topRows(filasTrain);
    /*
     * Seleccionadas las filas superiores para entrenamiento, se seleccionan las 11 primeras columnas
     * (columnas izquierdas) que representan las variables independientes FEATURES.
     */
    Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols()-1);

    /*
     * Se selecciona la variable dependiente que corresponde a la última columna.
     */
    Eigen::MatrixXd y_train = entrenamiento.rightCols(1);
    /*
     * Se realiza lo mismo para el conjunto de pruebas
     */

    Eigen::MatrixXd test = datos.bottomRows(filasTest);
    Eigen::MatrixXd X_test = test.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_test= test.rightCols(1);

    /*
     * Finalmente se retorna una tupla dada por el conjunto de datos de prueba y de entrenamiento.
     */

    return std::make_tuple(X_train,y_train,X_test,y_test);
}

/*
 * Se implementan 2 funciones para exportar a ficheros desde vector y desde eigen
 */
void ExEigenNorm::VectorToFile(std::vector<float> vector, std::string name){
    std::ofstream fichero(name);
    std::ostream_iterator<float> iterador(fichero, "\n");
    std::copy(vector.begin(),vector.end(),iterador);
}

void ExEigenNorm::EigenToFile(Eigen::MatrixXd datos, std::string name){
    std::ofstream fichero(name);
    if(fichero.is_open()){
        fichero<<datos<<"\n";
    }
}
