#pragma once

#include <iostream>
#include <fstream>
#include "ui_QtGuiApplication1.h"
#include <QtWidgets/QMainWindow>

#include "QPushButton"
#include"QSpinBox"

#include <QtWidgets/QMainWindow>
#include <QWidget>
#include <QAction>
#include <QFileDialog>
#include <QList>
#include <QString>
#include <QLabel>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QImageReader>
#include <QPixmap>
#include <QPainter>
#include<QScrollArea>
#include<stdio.h>
#include <QImage>

#include <QColor>
#include <QApplication>
#include <QMouseEvent>

#include <iostream>
#include <fstream>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <json.hpp>

#include "itkImageFileReader.h"
#include "itkGDCMImageIO.h"
#include <itkNiftiImageIO.h>
#include "itkGDCMSeriesFileNames.h"
#include "itkImageFileReader.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkMetaDataObject.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImageToVTKImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include <itkConnectedComponentImageFilter.h>
#include "itkScalarToRGBPixelFunctor.h"

#include "itkLabelMapOverlayImageFilter.h"

#include "itkConnectedThresholdImageFilter.h"


#include "itkNumericSeriesFileNames.h"
#include <QTextStream>


#include "itkImageRegionIteratorWithIndex.h"
#include<qtablewidget.h>
#include <itkLabelMap.h>
#include <itkLabelObject.h>
#include <itkBinaryImageToLabelMapFilter.h>
#include<qwaitcondition.h>
#include <itkOrientImageFilter.h>

#include "itkResampleImageFilter.h"

#include "itkIdentityTransform.h"

#include "vtkAutoInit.h"

#include "vtkMarchingCubes.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include <vtkWindowToImageFilter.h>
#include "vtkAutoInit.h" 
#include <itkVTKImageToImageFilter.h>
#include "itkOpenCVImageBridge.h"

#include <vtkDiscreteMarchingCubes.h>
#include "itkLabelImageToShapeLabelMapFilter.h"
#include <itkNearestNeighborInterpolateImageFunction.h>
//#include "vtkAutoInit.h" 

using namespace std;
using namespace cv;
using namespace itk;
using namespace vtk;


// ITK Definitions
const unsigned int Dimension3D = 3;
const unsigned int Dimension2D = 2;
const unsigned int OutputDimension =2;
using PixelTypeSC = signed short;
using PixelTypeUC = unsigned char;
using RGBPixelType = itk::RGBPixel<unsigned char>;
//using RGBPixelTypeSC = itk::RGBPixel< signed short>;
using RGBImageType = itk::Image<RGBPixelType, Dimension2D>;
using RGBSeriesType = itk::Image<RGBPixelType, Dimension3D>;
//using RGBImageTypeSC = itk::Image<RGBPixelTypeSC, Dimension2D>;
//using RGBSeriesTypeSC = itk::Image<RGBPixelTypeSC, Dimension3D>;
using OutputPixelType = unsigned char;

using Image2DType = itk::Image< OutputPixelType, OutputDimension >;




//using LabelType = unsigned char;
//using LabelObjectType = itk::LabelObject<LabelType, Dimension3D>;
using SeriesTypeSC = itk::Image<PixelTypeSC, Dimension3D>; //Serie DICOM originale
using SeriesTypeUC = itk::Image<PixelTypeUC, Dimension3D>; // Serie DICOM [0 255]


using ImageTypeUC = itk::Image<PixelTypeUC, Dimension2D>; // Immagine 2D Dicom
//using IteratorType = itk::ImageRegionIteratorWithIndex< ImageTypeUC >;
using ImageTypeSC = itk::Image<PixelTypeSC, Dimension2D>; // Immagine 2D Dicom
using LabelType = unsigned char;
using LabelObjectType = itk::LabelObject<LabelType, Dimension3D>;
using LabelMapType = itk::LabelMap<LabelObjectType>;


using ShapeLabelObjectType = itk::ShapeLabelObject<LabelType, Dimension3D>;
using LabelShapeType = itk::LabelMap<ShapeLabelObjectType>;


using IteratorType = itk::ImageRegionIteratorWithIndex< ImageTypeUC >;
using IteratorTypeRGB = itk::ImageRegionIteratorWithIndex< RGBImageType >;
using ImageReaderType = itk::ImageFileReader<SeriesTypeSC>;
using ImageReaderTypeUC = itk::ImageFileReader<SeriesTypeUC>;
using SeriesReaderType = itk::ImageSeriesReader<SeriesTypeSC>;
using SeriesWriterType = itk::ImageSeriesWriter<SeriesTypeUC, Image2DType>;
using TransformType = itk::IdentityTransform<double, Dimension3D>;
using InterpolatorType = itk::NearestNeighborInterpolateImageFunction<SeriesTypeUC ,double>;

class QtGuiApplication1 : public QMainWindow
{
	Q_OBJECT

public:
	QtGuiApplication1(QWidget *parent = Q_NULLPTR);
	void ShowDicomSeries(SeriesTypeUC::Pointer& in, QSpinBox* bar1, QSpinBox* bar2, QSpinBox* bar3, QLabel* label1, QLabel*label2,	QLabel* label3);
	bool readDICOMSeries(string series_path, SeriesReaderType::Pointer reader);
	void ShowMasks(SeriesTypeUC::Pointer& in, RGBSeriesType::Pointer& masks, QSpinBox* bar1, QSpinBox* bar2, QSpinBox* bar3, QLabel* label1, QLabel* label2, QLabel* label3);
	//void printDicomHeaderInfo(QTableWidget* table);
	//bool writeDICOMSeries(string seriesOutPath);
	void readJsonFile(std::string filename, vector<uchar>& label);
	void resampleSeries(SeriesTypeUC::Pointer & Seriesin);
	void fromLabeltoName(vector<uchar>& labels, vector<cv::String>& vertebraes);
private:
	Ui::QtGuiApplication1Class ui;
	void onTriggered(bool checked);
	bool onTriggered_close(bool checked);
	void onClicked_masks(bool checked);
	void onClicked_mesh(bool checked);
	bool eventFilter(QObject* obj, QEvent* ev);
	void onTriggered_learn(bool checked);

	bool close_window = false;
	//SeriesReaderType::Pointer reader;
	//SeriesReaderType::Pointer maskreader;
	SeriesTypeSC::Pointer processedSeries;
	SeriesTypeSC::Pointer processedSeries1;
	SeriesTypeUC::Pointer ucProcessedSeries;
	SeriesTypeUC::Pointer ucProcessedSeries1;
	SeriesTypeUC::Pointer mask4mesh;
	SeriesTypeUC::Pointer maskuc;
	RGBSeriesType::Pointer maskrgb;


	std::vector<uchar> labels;
	/*vector<string> vertebraes;*/
	vector<cv::String> vertebraes;
	vector<cv::Point> landmarks_sag;
	vector<cv::Point> landmarks_cor;


	
};
