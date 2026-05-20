# Descripción de la carpeta MARS

La carpeta `MARS/` contiene el dataset final de imágenes de Marte y máscaras de tormentas de polvo. Los archivos están organizados con rutas relativas, por lo que la carpeta puede usarse en otro computador siempre que se conserve su estructura interna.

## images/

Contiene las imágenes RGB de Marte.

Estas imágenes son la entrada del dataset. Representan mapas globales diarios de Marte y pueden usarse para visualización o como entrada de modelos de segmentación.

## masks_multiclass/

Contiene las máscaras multiclase asociadas a las imágenes RGB.

Cada máscara identifica tres tipos de píxeles:

- `0` = no tormenta
- `1` = tormenta de polvo
- `2` = no-data / dato no válido

Estas máscaras ya tienen corregida su orientación espacial respecto a las imágenes RGB, por lo que no deben rotarse nuevamente.

## mars_manifest_complete.csv

Archivo principal del dataset.

Relaciona cada imagen RGB con su máscara correspondiente mediante rutas relativas. Permite saber qué imagen de `images/` corresponde con qué máscara de `masks_multiclass/`.

También incluye información adicional como el índice temporal, el día marciano y el porcentaje de tormenta sobre píxeles válidos.

## MDADMY32.nc

Archivo original de la base MDAD para el año marciano MY32.

Contiene la información original de las máscaras de tormentas de polvo y zonas sin datos. Se conserva como archivo fuente para trazabilidad o para reconstruir nuevas máscaras si fuera necesario.

## mosaico_top17_rgb.png

Imagen de verificación que muestra las 17 imágenes RGB seleccionadas.

Su función es permitir una revisión visual rápida de las imágenes originales del dataset.

## mosaico_top17_rgb_overlay_multiclass.png

Imagen de verificación que muestra las 17 imágenes RGB con la máscara multiclase superpuesta.

Su función es comprobar visualmente que las regiones de tormenta y no-data estén correctamente alineadas con las imágenes.

## images_preparation.ipynb

Notebook usado para preparar el dataset.

Contiene el procesamiento realizado para organizar las imágenes, generar las máscaras multiclase, crear el archivo manifest y producir las figuras de verificación.

## Estado del dataset

Las imágenes RGB y sus máscaras multiclase ya están listas para usarse en otro código.

Para trabajar con el dataset completo se deben usar:

- `images/`
- `masks_multiclass/`
- `mars_manifest_complete.csv`

El archivo `mars_manifest_complete.csv` es el que asegura la correspondencia correcta entre cada imagen y su máscara.