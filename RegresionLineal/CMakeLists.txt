cmake_minimum_required(VERSION 3.5)

project(RegresionLineal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



#Se crea una lista con todos los .cpp añadidos al proyecto.
set(project_sources
    main.cpp
    exeigennorm.cpp
    linealregression.cpp)

#Se crea el ejecutable con todos los archivos fuentes necesarios.
add_executable(${PROJECT_NAME} ${project_sources})
