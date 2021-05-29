#include "QtGuiApplication1.h"
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
#include<QPushButton>
#include <QImage>
#include <QColor>
#include <QApplication>
#include <QTextStream>                                                
#include<qtablewidget.h>
#include<qwaitcondition.h>
#include <QDesktopServices>
#include <QMouseEvent>
#include<stdio.h>
#include <iostream>
#include <fstream>
#include "json.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "itkRGBPixel.h"
#include "itkImageFileReader.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include <itkNiftiImageIO.h>
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkMetaDataObject.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImageToVTKImageFilter.h"


#include <itkConnectedComponentImageFilter.h>
#include "itkScalarToRGBPixelFunctor.h"
#include "itkChangeInformationImageFilter.h"
#include "itkConnectedThresholdImageFilter.h"
#include "itkThresholdImageFilter.h"

#include "itkNumericSeriesFileNames.h"

#include"itkLabelImageToLabelMapFilter.h"

#include <itkLabelMap.h>
#include <itkLabelObject.h>
#include "itkLabelMapOverlayImageFilter.h"
#include <itkBinaryImageToLabelMapFilter.h>

#include"itkLabelSelectionLabelMapFilter.h"
#include"itkLabelMapToLabelImageFilter.h"
#include "itkLabelImageToShapeLabelMapFilter.h"
#include "itkOpenCVImageBridge.h"
#include <itkOrientImageFilter.h>
#include "vtkAutoInit.h"


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
#include <vtkPNGWriter.h>

#include <itkChangeLabelLabelMapFilter.h>
#include <vtkDiscreteMarchingCubes.h>
#include <vtkNamedColors.h>
#include <vtkLookupTable.h>
#include "itkCastImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"


VTK_MODULE_INIT(vtkRenderingOpenGL2); // VTK was built with vtkRenderingOpenGL2
VTK_MODULE_INIT(vtkInteractionStyle);


using namespace std;
using namespace itk;
using namespace cv;
using namespace vtk;


void imageWindowingITK(SeriesTypeSC::Pointer& in, SeriesTypeSC::Pointer& out, int ww, int wl);
void rescaleITK(SeriesTypeSC::Pointer& in, SeriesTypeUC::Pointer& out, int min, int max);
bool close_window = false;


QtGuiApplication1::QtGuiApplication1(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	ui.viewlabel_ax->installEventFilter(this);
	ui.viewlabel_sag->installEventFilter(this);
	ui.viewlabel_cor->installEventFilter(this);
	ui.menuFile->setToolTip("File");
	ui.menuHelp->setToolTip("Help");
	ui.show_masks->setToolTip("Show the results of the segmentation algorithm");
	ui.mesh_button->setToolTip("Show 3D volume reconstruction of segmentation masks");
	
	connect(ui.actionOpen_DICOM_Series, static_cast<void (QAction::*)(bool)>(&QAction::triggered), this, &QtGuiApplication1::onTriggered);
	connect(ui.actionClose_DICOM_Series, static_cast<void (QAction::*)(bool)>(&QAction::triggered), this, &QtGuiApplication1::onTriggered_close);
	connect(ui.actionLearn_more, static_cast<void (QAction::*)(bool)>(&QAction::triggered), this, &QtGuiApplication1::onTriggered_learn);
	connect(ui.show_masks, static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), this, &QtGuiApplication1::onClicked_masks);
	connect(ui.mesh_button, static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), this, &QtGuiApplication1::onClicked_mesh);

}




void QtGuiApplication1::resampleSeries(SeriesTypeUC::Pointer& Seriesin)
{

	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	TransformType::Pointer transform = TransformType::New();
	transform->SetIdentity();

	const SeriesTypeUC::SpacingType inputSpacing = Seriesin->GetSpacing();
	const SeriesTypeUC::RegionType inputRegion = Seriesin->GetLargestPossibleRegion();
	const SeriesTypeUC::SizeType inputSize = inputRegion.GetSize();

	SeriesTypeUC::SpacingType outputSpacing;
	outputSpacing[0] = inputSpacing[0];
	outputSpacing[1] = inputSpacing[1];
	outputSpacing[2] = inputSpacing[2]*inputSize[2] / inputSize[0];

	bool changeInSpacing = false;
	for (unsigned int i = 0; i < 3; i++)
	{
		if (outputSpacing[i] == 0.0)
		{
			outputSpacing[i] = inputSpacing[i];
		}
		else
		{
			changeInSpacing = true;
		}
	}
	SeriesTypeUC::SizeType outputSize = inputSize;
	//outputSize[2] = inputSize[2] / (outputSpacing[2] / inputSpacing[2]);
	outputSize[2] = static_cast<SizeValueType>(inputSize[2] * inputSpacing[2] / outputSpacing[2]);
	using ResampleFilterType = itk::ResampleImageFilter<SeriesTypeUC, SeriesTypeUC>;
	ResampleFilterType::Pointer resampler = ResampleFilterType::New();
	resampler->SetInput(Seriesin);
	resampler->SetTransform(transform);
	resampler->SetInterpolator(interpolator);
	resampler->SetOutputOrigin(Seriesin->GetOrigin());
	resampler->SetOutputSpacing(outputSpacing);
	resampler->SetOutputDirection(Seriesin->GetDirection());
	resampler->SetSize(outputSize);
	resampler->Update();
	Seriesin = resampler->GetOutput();
}


void QtGuiApplication1::onClicked_mesh(bool checked) {
	//vtkObject::GlobalWarningDisplayOff();

	using itkVtkConverter2 = itk::ImageToVTKImageFilter<SeriesTypeUC>;
		itkVtkConverter2::Pointer connector2 = itkVtkConverter2::New();
		connector2->SetInput(maskuc);
		connector2->Update();
	
		vtkSmartPointer<vtkDiscreteMarchingCubes> cube2 = vtkDiscreteMarchingCubes::New();
		cube2->SetInputData(connector2->GetOutput());
	
		//cube2->SetValue(0, 26);
		//cube2->ComputeGradientsOn();
		//cube2->ComputeNormalsOn();
		
		cube2->ComputeAdjacentScalarsOn();
		cube2->ComputeScalarsOn();
		//cube2->GenerateValues(4, 1, 33);
	
		cube2->Update();
	
		//int r =  255;
		//double rd = (double)r / 255;
		//int b = 0;
		//double bd = (double)b / 255;
		//int g = 0;
		//double gd = (double)g / 255;
		//int re = rand() % 255;
		//

		

		vtkSmartPointer <vtkPolyDataMapper> mapper2 = vtkPolyDataMapper::New();
		//mapper->SetInputConnection(cube->GetOutputPort());
		mapper2->SetInputConnection(cube2->GetOutputPort());
		//mapper2->SetScalarRange(0.3, 1);
		mapper2->ScalarVisibilityOff();
		
		//mapper2->SetScalarModeToUsePointData();
		//mapper2->SetLookupTable(lut);
		vtkSmartPointer < vtkActor > actor2 = vtkActor::New();
		actor2->SetMapper(mapper2);

		//actor2->GetProperty()->SetColor(rd,gd,bd);
		actor2->GetProperty()->SetOpacity(1);
		actor2->RotateX(90);
		//actor2->RotateZ(180);
		actor2->ApplyProperties();
		
		cube2->Delete();
		mapper2->Delete();
		actor2->Delete();
	
		vtkSmartPointer<vtkRenderer> renderer2 = vtkSmartPointer<vtkRenderer>::New();
		renderer2->AddActor(actor2);
		renderer2->SetBackground(0, 0, 0);
		renderer2->ResetCamera();
		vtkSmartPointer<vtkRenderWindow> renWin2 = vtkSmartPointer<vtkRenderWindow>::New();
		
		renWin2->AddRenderer(renderer2);
		renWin2->SetOffScreenRendering(1);
		
		renWin2->Render(); // make sure we have an OpenGL context.
	

		vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =vtkSmartPointer<vtkWindowToImageFilter>::New();
		windowToImageFilter->SetInput(renWin2);
		windowToImageFilter->Update();


		vtkSmartPointer<vtkPNGWriter> writer =
			vtkSmartPointer<vtkPNGWriter>::New();
		writer->SetFileName("mesh.png");
		writer->SetInputConnection(windowToImageFilter->GetOutputPort());
		writer->Write();

		using ToITKFilterType = itk::VTKImageToImageFilter<RGBImageType>;
		ToITKFilterType::Pointer toitkfilter = ToITKFilterType::New();
		toitkfilter->SetInput(windowToImageFilter->GetOutput());
		toitkfilter->Update();

		Mat meshImg = itk::OpenCVImageBridge::ITKImageToCVMat<RGBImageType>(toitkfilter->GetOutput());
		cvtColor(meshImg, meshImg, COLOR_RGB2BGR);
	
		QImage mesh((const uchar*)meshImg.data, meshImg.cols, meshImg.rows, static_cast<int>(meshImg.step), QImage::Format_RGB888);
		//assiale = Mat2QImage(axialImg);
		QPixmap imgmesh = QPixmap::fromImage(mesh, Qt::ColorOnly);
		ui.viewlabel_mesh->setPixmap(imgmesh);


}

void QtGuiApplication1::onTriggered_learn(bool checked) {
	QDesktopServices::openUrl(QUrl("https://www.researchgate.net/publication/338853005_VerSe_A_Vertebrae_Labelling_and_Segmentation_Benchmark"));
}


void QtGuiApplication1::onClicked_masks(bool checked) {
	ui.mesh_button->setEnabled(true);
	//non_devo_uscire = true;
	//QString fileName = QFileDialog::getExistingDirectory(this, tr("Path"));   //getOpenFileName(this, tr("Image Files "));
	//SeriesReaderType::Pointer maskreader = SeriesReaderType::New();
	//ImageReaderType::Pointer maskreader = ImageReaderType::New();

	//LEGGO FILE JSON
	std::string JsonFilename = "landmark.json";


	//SE LEGGO NIFTI UC
	QString fileName = QFileDialog::getOpenFileName(this, tr("Image Files"));	
	ImageReaderTypeUC::Pointer maskreader = ImageReaderTypeUC::New();
	std::string series_path = fileName.toLocal8Bit().constData();
	//maskreader = SeriesReaderType::New();
	itk::NiftiImageIO::Pointer niftiIO = itk::NiftiImageIO::New();
	maskreader->SetImageIO(niftiIO);
	maskreader->SetFileName(series_path);
	maskreader->Update();

	const SeriesTypeUC::SpacingType inputSpacing = maskreader->GetOutput()->GetSpacing();
	const SeriesTypeUC::RegionType inputRegion = maskreader->GetOutput()->GetLargestPossibleRegion();
	const SeriesTypeUC::SizeType inputSize = inputRegion.GetSize();


	itk::OrientImageFilter<SeriesTypeUC, SeriesTypeUC>::Pointer orienter =
		itk::OrientImageFilter<SeriesTypeUC, SeriesTypeUC>::New();
	orienter->UseImageDirectionOn();
	orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
	orienter->SetInput(maskreader->GetOutput());
	orienter->Update();

	maskuc = SeriesTypeUC::New();
	maskuc = orienter->GetOutput();
	resampleSeries(maskuc);
	
	

	using LabelImageToLabelMapFilterType = itk::LabelImageToLabelMapFilter<SeriesTypeUC>;
	LabelImageToLabelMapFilterType::Pointer labelImageToLabelMapFilter = LabelImageToLabelMapFilterType::New();
	labelImageToLabelMapFilter->SetInput(maskuc);
	labelImageToLabelMapFilter->Update();
	labelImageToLabelMapFilter->GetOutput()->PrintLabelObjects();

	


	using LabelMapOverlayImageFilterType =
		itk::LabelMapOverlayImageFilter<LabelMapType, SeriesTypeUC, RGBSeriesType>;
	LabelMapOverlayImageFilterType::Pointer labelMapOverlayImageFilter = LabelMapOverlayImageFilterType::New();
	labelMapOverlayImageFilter->SetInput(labelImageToLabelMapFilter->GetOutput());
	labelMapOverlayImageFilter->SetFeatureImage(maskuc);
	labelMapOverlayImageFilter->SetOpacity(.5);
	labelMapOverlayImageFilter->Update();

	
	using I2LType = itk::LabelImageToShapeLabelMapFilter<SeriesTypeUC, LabelShapeType>;

	I2LType::Pointer i2l = I2LType::New();
	i2l->SetInput(maskuc);
	//i2l->SetComputePerimeter(true);
	i2l->Update();
	
	readJsonFile("landmark.json", labels);
	
	for ( int n = 0; n < labels.size(); ++n)
	{
		
		
		ShapeLabelObjectType* labelObject = i2l->GetOutput()->GetNthLabelObject(n);
		
		std::cout << "    Centroid: " << labelObject->GetCentroid() << std::endl;
		double x = labelObject->GetCentroid()[1];
		double y = labelObject->GetCentroid()[2];
		double z = labelObject->GetCentroid()[0];
		landmarks_sag.push_back(cv::Point(x*2,inputSize[0]-y*inputSize[0]/(inputSize[2]*inputSpacing[2])));
		landmarks_cor.push_back(cv::Point(z * 2, inputSize[0] - y * inputSize[0] / (inputSize[2] * inputSpacing[2])));
		cout << landmarks_sag[n];
	}


	maskrgb = RGBSeriesType::New();
	maskrgb = labelMapOverlayImageFilter->GetOutput();

	ShowMasks(ucProcessedSeries, maskrgb, ui.slicenumber_scroll_ax, ui.slicenumber_scroll_sag, ui.slicenumber_scroll_cor, ui.viewlabel_ax, ui.viewlabel_sag, ui.viewlabel_cor);



}




bool QtGuiApplication1::onTriggered_close(bool checked) {
	close_window = true;
	
	
	ui.viewlabel_ax->clear();
	ui.viewlabel_sag->clear();
	ui.viewlabel_cor->clear();
	ui.viewlabel_mesh->clear();
	
	return true;
}




void QtGuiApplication1::onTriggered(bool checked) {
	//non_devo_uscire = true;
	
	//SE DEVO LEGGERE DICOM
	QString fileName = QFileDialog::getExistingDirectory(this, tr("Path"));   //getOpenFileName(this, tr("Image Files "));
	std::string series_path = fileName.toLocal8Bit().constData();
	SeriesReaderType::Pointer reader = SeriesReaderType::New();
	readDICOMSeries(series_path, reader);


	//SE DEVO LEGGERE NIFTI
	//QString fileName = QFileDialog::getOpenFileName(this, tr("Image Files" ));
	//std::string series_path = fileName.toLocal8Bit().constData();
	//ImageReaderType::Pointer reader_nifti = ImageReaderType::New();
	//itk::NiftiImageIO::Pointer niftiIO = itk::NiftiImageIO::New();
	//reader_nifti->SetImageIO(niftiIO);
	//reader_nifti->SetFileName(series_path);
	//reader_nifti->Update();
	//printDicomHeaderInfo(ui.tableWidget);

	itk::OrientImageFilter<SeriesTypeSC, SeriesTypeSC>::Pointer orienter =
		itk::OrientImageFilter<SeriesTypeSC, SeriesTypeSC>::New();
	orienter->UseImageDirectionOn();
	orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
	orienter->SetInput(reader->GetOutput());
	orienter->Update();
	
	

	int ww, wl;
	ww = 2000;
	wl = 800;



	processedSeries = SeriesTypeSC::New();
	//processedSeries = reader_nifti->GetOutput();
	processedSeries = orienter->GetOutput();
	//resampleSeries(processedSeries);

	imageWindowingITK(processedSeries, processedSeries, ww, wl);
	ucProcessedSeries = SeriesTypeUC::New();
	rescaleITK(processedSeries, ucProcessedSeries, 0, 255);

	resampleSeries(ucProcessedSeries);
	ShowDicomSeries(ucProcessedSeries, ui.slicenumber_scroll_ax, ui.slicenumber_scroll_sag, ui.slicenumber_scroll_cor, ui.viewlabel_ax, ui.viewlabel_sag,ui.viewlabel_cor);
	/*
	rescaleITK(processedSeries1, ucProcessedSeries1, 0, 255);*/
	while (waitKey(20) != ' ' && ui.windowing_check->isChecked() && close_window== false) {

		SeriesTypeSC::Pointer processedSeries1 = SeriesTypeSC::New();
		processedSeries1 = orienter->GetOutput();

		SeriesTypeUC::Pointer ucProcessedSeries1 = SeriesTypeUC::New();
		ww = ui.ww_spinbox->value();
		wl = ui.wl_spinbox->value();




		imageWindowingITK(processedSeries1, processedSeries1, ww, wl);
		rescaleITK(processedSeries1, ucProcessedSeries1, 0, 255);
		resampleSeries(ucProcessedSeries1);
		ShowDicomSeries(ucProcessedSeries1, ui.slicenumber_scroll_ax, ui.slicenumber_scroll_sag, ui.slicenumber_scroll_cor, ui.viewlabel_ax, ui.viewlabel_sag, ui.viewlabel_cor);

	}




}


bool  QtGuiApplication1::readDICOMSeries(const std::string seriesPath, SeriesReaderType::Pointer reader)
{
	// Imposto image IO per specificare il tipo di file da leggere (GDCM è progetto opensource DICOM)
	using ImageIO = itk::GDCMImageIO;
	ImageIO::Pointer dicomIO = ImageIO::New();
	reader->SetImageIO(dicomIO);

	//serie dicom  costituita da una sequenza di file
	// Genero i nomi dei file: Posso dire a itk di generare la lista dei file da leggere in base a quelli presenti in una cartella
	using NameGeneratorType = itk::GDCMSeriesFileNames;//questa classe ha un metodo, che data la cartella, genera una
	//lista di nomi che corrispondono a tt le immagini dicom presenti in cartella
	NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();
	nameGenerator->SetInputDirectory(seriesPath);
	nameGenerator->Update();

	vector<string> UIDs = nameGenerator->GetSeriesUIDs(); //data la cartella si prende tt gli UID presenti in cartella
	if (UIDs.size() == 0)
		return false;
	for (int i = 0; i < UIDs.size(); i++)
		std::cout << i << ": " << UIDs[i] << std::endl;

	int seriesIDX = 0;

	//cout << "Inserisci indice della serie: ";


	//se cartella con le immagini contiene + acquisizioni (di diverso tipo) , i file sono in un unica cartella 
	//e se voglio quelli di una determinata serie devo risalire all'UID della serie 
	reader->SetFileNames(nameGenerator->GetFileNames(UIDs[seriesIDX])); //lo parametrizzo con UID della serie
	//recupero i file con l'UID della serie che voglio
	//reader->SetReverseOrder(true);
	// Le righe precedenti sono equivalenti alla seguente:
	// ImageFileReader< Image< unsigned short, 2 > >::Pointer reader = ImageFileReader< Image< unsigned short, 2 > >::New();

	try {


		// Dopo aver impostato i parametri, è necessario chiamare Update() per eseguire effettivamente l'operazione
		reader->Update();
		//printDicomHeaderInfo(reader);



	}
	catch (itk::ExceptionObject & error) {
		std::cerr << "Error: " << error << std::endl;
		return false;
	}


	return true;
}
//bool QtGuiApplication1::writeDICOMSeries(string seriesOutPath)
//{
//	using ImageIOType = itk::GDCMImageIO;
//	//using NamesGeneratorType = itk::GDCMSeriesFileNames;
//	using NamesGeneratorType = itk::NumericSeriesFileNames;
//	ImageIOType::Pointer gdcmIO = ImageIOType::New();
//	NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
//	SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
//
//	seriesWriter->SetInput(ucProcessedSeries);
//	seriesWriter->SetImageIO(gdcmIO);
//	SeriesTypeUC::RegionType region =
//		reader->GetOutput()->GetLargestPossibleRegion();
//
//	SeriesTypeUC::IndexType start = region.GetIndex();
//	SeriesTypeUC::SizeType  size = region.GetSize();
//	namesGenerator->SetStartIndex(start[2]);
//	namesGenerator->SetEndIndex(start[2] + size[2] - 1);
//	namesGenerator->SetIncrementIndex(1);
//
//
//	std::string format = seriesOutPath;
//
//	format += "/image%03d.dcm";
//
//	namesGenerator->SetSeriesFormat(format.c_str());
//
//
//	/*namesGenerator->SetOutputDirectory(seriesOutPath);
//	namesGenerator->Update();*/
//	seriesWriter->SetFileNames(namesGenerator->GetFileNames());
//	//seriesWriter->SetMetaDataDictionaryArray(reader->GetMetaDataDictionaryArray());
//	try
//	{
//		seriesWriter->Update();
//	}
//	catch (itk::ExceptionObject & error)
//	{
//		std::cerr << "Error: " << error << std::endl;
//		return false;
//	}
//	return true;
//
//}
//
//void QtGuiApplication1::printDicomHeaderInfo(QTableWidget* table)
//{
//	// Stampa header info
//	using DictionaryType = itk::MetaDataDictionary;
//	DictionaryType& dictionary = reader->GetImageIO()->GetMetaDataDictionary();
//	using MetaDataStringType = itk::MetaDataObject<std::string>;
//	DictionaryType::Iterator itr = dictionary.Begin();
//	DictionaryType::Iterator end = dictionary.End();
//	int i = 0;
//
//	while (itr != end) {
//		itk::MetaDataObjectBase::Pointer entry = itr->second;
//		MetaDataStringType::Pointer entryvalue = dynamic_cast<MetaDataStringType*>(entry.GetPointer());
//		if (entryvalue) {
//			string tagkey = itr->first;
//			string labelId;
//			bool found = itk::GDCMImageIO::GetLabelFromTag(tagkey, labelId);    // Se non trova corrispondenza mette Unknown
//			string tagvalue = entryvalue->GetMetaDataObjectValue();
//			QString qtagkey = QString::fromStdString(tagkey);
//			QString qlabelId = QString::fromStdString(labelId);
//			QString qtagvalue = QString::fromStdString(tagvalue);
//
//
//			table->setItem(i, 0, new QTableWidgetItem(qtagkey));
//			table->setItem(i, 1, new QTableWidgetItem(qlabelId));
//			table->setItem(i, 2, new QTableWidgetItem(qtagvalue));
//
//
//
//		}
//		++itr;
//		i += 1;
//	}
//
//}




void imageWindowingITK(SeriesTypeSC::Pointer& in, SeriesTypeSC::Pointer& out, int ww, int wl)
{
	// Definizione del filtro (questo tipo di filtro vuole sapere che tipo di immagine è in ingresso e che tipo deve restituire (in generale tutti i filtri hanno questa forma))
	using WindowingFilterType = itk::IntensityWindowingImageFilter<SeriesTypeSC, SeriesTypeSC>;
	WindowingFilterType::Pointer windower = WindowingFilterType::New();
	// NB. ww e wl devo essere dello stesso tipo dei pixel (o cmq tpi compatibili)
	windower->SetInput(in);
	windower->SetWindowLevel(ww, wl);
	// Chiamata alla funzione Update per eseguire il filtro
	windower->Update();

	// Prendo il risultato
	out = windower->GetOutput();
}

void rescaleITK(SeriesTypeSC::Pointer& in, SeriesTypeUC::Pointer& out, int min, int max)
{
	// Definizione del filtro (questo tipo di filtro vuole sapere che tipo di immagine è in ingresso e che tipo deve restituire (in generale tutti i filtri hanno questa forma))
	using RescalingFilterType = itk::RescaleIntensityImageFilter<SeriesTypeSC, SeriesTypeUC>;
	RescalingFilterType::Pointer rescaler = RescalingFilterType::New();
	// NB. ww e wl devo essere dello stesso tipo dei pixel (o cmq tpi compatibili)
	rescaler->SetInput(in);
	rescaler->SetOutputMinimum(min);
	rescaler->SetOutputMaximum(max);
	// Chiamata alla funzione Update per eseguire il filtro
	rescaler->Update();

	// Prendo il risultato
	out = rescaler->GetOutput();
}



void QtGuiApplication1::ShowDicomSeries(SeriesTypeUC::Pointer& in, QSpinBox* bar_ax, QSpinBox* bar_sag, QSpinBox* bar_cor, QLabel* label_ax, QLabel* label_sag, QLabel* label_cor)
{

	SeriesTypeUC::RegionType inputRegion = in->GetLargestPossibleRegion();
	SeriesTypeUC::SizeType regionSize = inputRegion.GetSize(); //restituisce un vettore cui dimensioni sono quelle della serie nelle 3 viste
	int numSlicesSagittal = regionSize[0];
	int numSlicesCoronal = regionSize[1];
	int numSlicesAxial = regionSize[2];

	regionSize[1] = 0; //PER ESTRARRE UN'IMMAGINE SI IMPOSTA A 0 L'INDICE DELLA REGIONE DI CUI VOGLIO PRENDERE L'IMMAGINE

	/* https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch1.html
	Then, we take the index from the region and set its Z value to the slice number we want to  extract.
	*/
	//Creo una nuova regione parametrizzata con la slice che voglio prendere

	int sliceNumber = 250;
	SeriesTypeUC::IndexType start = inputRegion.GetIndex();
	start[1] = sliceNumber;

	/*Finally, an itk::ImageRegion object is created and initialized with the start and size we just
	prepared using the slice information.*/
	SeriesTypeUC::RegionType desiredRegion;
	desiredRegion.SetSize(regionSize);
	desiredRegion.SetIndex(start);


	using FilterType = itk::ExtractImageFilter<SeriesTypeUC, ImageTypeUC >;
	FilterType::Pointer extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();

	/* Costruiamo Visualizzatore DICOM */
	//FACCIO IL FILTRO PER ESTRARRE IL VOI:VOLUME OF INTEREST
	int axialInfo[10] = { 0, numSlicesAxial, 0, 0, numSlicesSagittal - 1, numSlicesCoronal - 1 , 0, 0,0,0 };
	int sagittalInfo[6] = { 0, numSlicesSagittal, 0, 0, numSlicesCoronal - 1, numSlicesAxial - 1 };
	int coronalInfo[6] = { 0, numSlicesCoronal, 0, 0, numSlicesSagittal - 1, numSlicesAxial - 1 };


	SeriesTypeUC::SizeType axialSize = inputRegion.GetSize();
	axialSize[2] = 0;

	SeriesTypeUC::SizeType coronalSize = inputRegion.GetSize();
	coronalSize[1] = 0;

	SeriesTypeUC::SizeType sagittalSize = inputRegion.GetSize();
	sagittalSize[0] = 0;

	SeriesTypeUC::IndexType axindex = inputRegion.GetIndex();
	SeriesTypeUC::IndexType corindex = inputRegion.GetIndex();
	SeriesTypeUC::IndexType sagindex = inputRegion.GetIndex();
	axindex[2] = 0;
	corindex[1] = 0;
	sagindex[0] = 0;


	 //ASSIALE
	desiredRegion.SetSize(axialSize);
	desiredRegion.SetIndex(axindex);

	extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();

	
	Mat axialImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
	QImage assiale((const uchar*)axialImg.data, axialImg.cols, axialImg.rows, static_cast<int>(axialImg.step), QImage::Format_Grayscale8);
	QPixmap imgassiale = QPixmap::fromImage(assiale, Qt::ColorOnly);
	label_ax->setPixmap(imgassiale);



	//CORONALE
	desiredRegion.SetSize(coronalSize);
	desiredRegion.SetIndex(corindex);

	extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();
	Mat corImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
	QImage coronale((const uchar*)corImg.data, corImg.cols, corImg.rows, static_cast<int>(corImg.step), QImage::Format_Grayscale8);
	
	QPixmap imgcoronale = QPixmap::fromImage(coronale, Qt::ColorOnly);
	
	label_cor->setPixmap(imgcoronale.scaled(label_cor->width(), label_cor->height(), Qt::KeepAspectRatio));

	//SAGITTALE
	desiredRegion.SetSize(sagittalSize);
	desiredRegion.SetIndex(sagindex);

	extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();
	Mat sagImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
	QImage sagittale((const uchar*)sagImg.data, sagImg.cols, sagImg.rows, static_cast<int>(sagImg.step), QImage::Format_Grayscale8);
	//assiale = Mat2QImage(axialImg);
	QPixmap imgsagittale = QPixmap::fromImage(sagittale, Qt::ColorOnly);
	label_sag->setPixmap(imgsagittale.scaled(label_sag->width(), label_sag->height(), Qt::KeepAspectRatio));
	//label2->adjustSize();

	bar_ax->setMaximum(numSlicesAxial - 1);
	bar_sag->setMaximum(numSlicesSagittal - 1);
	bar_cor->setMaximum(numSlicesCoronal - 1);
		do {
			//ASSIALE
			axialInfo[0] = bar_ax->value();
			axindex[2] = axialInfo[0];
			desiredRegion.SetSize(axialSize);
			desiredRegion.SetIndex(axindex);
			extractFilter = FilterType::New();
			extractFilter->SetExtractionRegion(desiredRegion);
			extractFilter->SetInput(in);
			extractFilter->SetDirectionCollapseToIdentity();
			extractFilter->Update();
			Mat axialImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
			QImage assiale((const uchar*)axialImg.data, axialImg.cols, axialImg.rows, static_cast<int>(axialImg.step), QImage::Format_Grayscale8);
			//assiale = Mat2QImage(axialImg);
			QPixmap imgassiale = QPixmap::fromImage(assiale, Qt::ColorOnly);
			label_ax->setPixmap(imgassiale);

			//CORONALE

			coronalInfo[0] = bar_cor->value();
			corindex[1] = coronalInfo[0];
			desiredRegion.SetSize(coronalSize);
			desiredRegion.SetIndex(corindex);

			extractFilter = FilterType::New();
			extractFilter->SetExtractionRegion(desiredRegion);
			extractFilter->SetInput(in);
			extractFilter->SetDirectionCollapseToIdentity();
			extractFilter->Update();
			
			Mat corImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
			QImage coronale((const uchar*)corImg.data, corImg.cols, corImg.rows, static_cast<int>(corImg.step), QImage::Format_Grayscale8);
			//assiale = Mat2QImage(axialImg);
			QPixmap imgcoronale = QPixmap::fromImage(coronale, Qt::ColorOnly);
			label_cor->setPixmap(imgcoronale.scaled(label_cor->width(), label_cor->height(), Qt::KeepAspectRatio));
		

			//SAGITTALE
			sagittalInfo[0] = bar_sag->value();
			sagindex[0] = sagittalInfo[0];
			desiredRegion.SetSize(sagittalSize);
			desiredRegion.SetIndex(sagindex);

			extractFilter = FilterType::New();
			extractFilter->SetExtractionRegion(desiredRegion);
			extractFilter->SetInput(in);
			extractFilter->SetDirectionCollapseToIdentity();
			extractFilter->Update();
			Mat sagImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
			QImage sagittale((const uchar*)sagImg.data, sagImg.cols, sagImg.rows, static_cast<int>(sagImg.step), QImage::Format_Grayscale8);
			//assiale = Mat2QImage(axialImg);
			QPixmap imgsagittale = QPixmap::fromImage(sagittale, Qt::ColorOnly);
			//label2->setPixmap(imgsagittale);
			label_sag->setPixmap(imgsagittale.scaled(label_sag->width(), label_sag->height(), Qt::KeepAspectRatio));
			//label2->adjustSize();
	
	
	
		} while (waitKey(100) != ' ' && ui.windowing_check->isChecked() == false && close_window== false);

		
}






bool QtGuiApplication1::eventFilter(QObject* obj, QEvent* ev) {
	if (ev->type() == QEvent::MouseButtonPress) {

		QMouseEvent* mouseev = static_cast<QMouseEvent*>(ev);
		QPoint P = mouseev->pos();
		QImage assiale = ui.viewlabel_ax->pixmap()->toImage();
		ui.intensity_edit->setText(QString::number(QColor(assiale.pixel(P.x() , P.y() )).red())); //la Qlabel è ha dimensioni 256x256 in modo tale da poter risalire alla poszione reale
		QImage sagittale = ui.viewlabel_sag->pixmap()->toImage();
		ui.intensity_edit->setText(QString::number(QColor(sagittale.pixel(P.x() , P.y() )).red())); 
		QImage coronale = ui.viewlabel_cor->pixmap()->toImage();
		ui.intensity_edit->setText(QString::number(QColor(coronale.pixel(P.x() , P.y())).red())); 
	
	
	}

	return false;
}


void QtGuiApplication1::readJsonFile(std::string filename, std::vector<uchar>& label)
{

	std::ifstream file(filename);
	nlohmann::json x;
	file >> x;
	std::cout <<x["number"] << std::endl;
	
	int num = x["number"];
	
	for (int i = 0;i < num;i++){
		label.push_back(uchar(x["vertebrae"][i]["label"]));
		//punti.push_back(cv::Point(x["vertebrae"][i]["X"], x["vertebrae"][i]["Y"]));
		cout << label[i]  ;
	}
	//cout << landmarks << endl;
}

void QtGuiApplication1::ShowMasks(SeriesTypeUC::Pointer& in, RGBSeriesType::Pointer& masks, QSpinBox* bar_ax, QSpinBox* bar_sag, QSpinBox* bar_cor, QLabel* label_ax, QLabel* label_sag, QLabel* label_cor)
{
	SeriesTypeUC::RegionType inputRegion = in->GetLargestPossibleRegion();
	SeriesTypeUC::SizeType regionSize = inputRegion.GetSize(); //restituisce un vettore cui dimensioni sono quelle della serie nelle 3 viste
	int numSlicesSagittal = regionSize[0];
	int numSlicesCoronal = regionSize[1];
	int numSlicesAxial = regionSize[2];

	regionSize[1] = 0; //PER ESTRARRE UN'IMMAGINE SI IMPOSTA A 0 L'INDICE DELLA REGIONE DI CUI VOGLIO PRENDERE L'IMMAGINE

	/* https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch1.html
	Then, we take the index from the region and set its Z value to the slice number we want to  extract.
	*/
	//Creo una nuova regione parametrizzata con la slice che voglio prendere

	int sliceNumber = 250;
	SeriesTypeUC::IndexType start = inputRegion.GetIndex();
	start[1] = sliceNumber;

	/*Finally, an itk::ImageRegion object is created and initialized with the start and size we just
	prepared using the slice information.*/
	SeriesTypeUC::RegionType desiredRegion;
	desiredRegion.SetSize(regionSize);
	desiredRegion.SetIndex(start);


	using FilterType = itk::ExtractImageFilter<SeriesTypeUC, ImageTypeUC >;
	FilterType::Pointer extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();

	

	

	/* Costruiamo Visualizzatore DICOM */
	//FACCIO IL FILTRO PER ESTRARRE IL VOI:VOLUME OF INTEREST
	int axialInfo[10] = { 0, numSlicesAxial, 0, 0, numSlicesSagittal - 1, numSlicesCoronal - 1 , 0, 0,0,0 };
	int sagittalInfo[6] = { 0, numSlicesSagittal, 0, 0, numSlicesCoronal - 1, numSlicesAxial - 1 };
	int coronalInfo[6] = { 0, numSlicesCoronal, 0, 0, numSlicesSagittal - 1, numSlicesAxial - 1 };


	SeriesTypeUC::SizeType axialSize = inputRegion.GetSize();
	axialSize[2] = 0;

	SeriesTypeUC::SizeType coronalSize = inputRegion.GetSize();
	coronalSize[1] = 0;

	SeriesTypeUC::SizeType sagittalSize = inputRegion.GetSize();
	sagittalSize[0] = 0;

	SeriesTypeUC::IndexType axindex = inputRegion.GetIndex();
	SeriesTypeUC::IndexType corindex = inputRegion.GetIndex();
	SeriesTypeUC::IndexType sagindex = inputRegion.GetIndex();
	axindex[2] = 0;
	corindex[1] = 0;
	sagindex[0] = 0;

	desiredRegion.SetSize(axialSize);
	desiredRegion.SetIndex(axindex);

	extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();

	Mat axialImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
	QImage assiale((const uchar*)axialImg.data, axialImg.cols, axialImg.rows, static_cast<int>(axialImg.step), QImage::Format_Grayscale8);
	//assiale = Mat2QImage(axialImg);
	QPixmap imgassiale = QPixmap::fromImage(assiale, Qt::ColorOnly);
	label_ax->setPixmap(imgassiale);



	//CORONALE
	desiredRegion.SetSize(coronalSize);
	desiredRegion.SetIndex(corindex);

	extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();
	Mat corImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
	QImage coronale((const uchar*)corImg.data, corImg.cols, corImg.rows, static_cast<int>(corImg.step), QImage::Format_Grayscale8);
	//assiale = Mat2QImage(axialImg);
	QPixmap imgcoronale = QPixmap::fromImage(coronale, Qt::ColorOnly);
	/*imshow("immagine Coronale", img);*/
	label_cor->setPixmap(imgcoronale.scaled(label_cor->width(), label_cor->height(), Qt::KeepAspectRatio));



	//SAGITTALE
	desiredRegion.SetSize(sagittalSize);
	desiredRegion.SetIndex(sagindex);

	extractFilter = FilterType::New();
	extractFilter->SetExtractionRegion(desiredRegion);
	extractFilter->SetInput(in);
	extractFilter->SetDirectionCollapseToIdentity();
	extractFilter->Update();
	Mat sagImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
	QImage sagittale((const uchar*)sagImg.data, sagImg.cols, sagImg.rows, static_cast<int>(sagImg.step), QImage::Format_Grayscale8);
	//assiale = Mat2QImage(axialImg);
	QPixmap imgsagittale = QPixmap::fromImage(sagittale, Qt::ColorOnly);
	
	label_sag->setPixmap(imgsagittale.scaled(label_sag->width(),label_sag->height(),Qt::KeepAspectRatio));
	

	bar_ax->setMaximum(numSlicesAxial - 1);
	bar_sag->setMaximum(numSlicesSagittal - 1);
	bar_cor->setMaximum(numSlicesCoronal - 1);
	
	Mat maskImgax, maskImgsag, maskImgcor;
	fromLabeltoName(labels, vertebraes);
	
	do {
		axialInfo[0] = bar_ax->value();
		axindex[2] = axialInfo[0];
		desiredRegion.SetSize(axialSize);
		desiredRegion.SetIndex(axindex);
		extractFilter = FilterType::New();
		extractFilter->SetExtractionRegion(desiredRegion);
		extractFilter->SetInput(in);
		extractFilter->SetDirectionCollapseToIdentity();
		extractFilter->Update();

		axialImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());
		
		using rgbFilterType = itk::ExtractImageFilter<RGBSeriesType, RGBImageType >;
		rgbFilterType::Pointer rgbextractFilter = rgbFilterType::New();
		rgbextractFilter = rgbFilterType::New();
		rgbextractFilter->SetExtractionRegion(desiredRegion);
		rgbextractFilter->SetInput(masks);
		rgbextractFilter->SetDirectionCollapseToIdentity();
		rgbextractFilter->Update();
		maskImgax = itk::OpenCVImageBridge::ITKImageToCVMat<RGBImageType>(rgbextractFilter->GetOutput());


		cvtColor(axialImg, axialImg, COLOR_GRAY2BGR);
		cvtColor(maskImgax, maskImgax, COLOR_RGB2BGR);

		addWeighted(axialImg, 0.5, maskImgax, 0.5, 0.0, axialImg);

		//for (int i = 0; i < landmarks.size();i++) {

		//	//circle(sagImg, landmarks[i], 10, Scalar(0, 0, 255), 3);
		//	putText(axialImg, vertebraes[i], landmarks[i], cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
		//}
		QImage assiale((const uchar*)axialImg.data, axialImg.cols, axialImg.rows, static_cast<int>(axialImg.step), QImage::Format_RGB888);
		//assiale = Mat2QImage(axialImg);
		QPixmap imgassiale = QPixmap::fromImage(assiale, Qt::ColorOnly);
		//imshow("Assiale1", axialImg);
		label_ax->setPixmap(imgassiale);


		//SAGITTALE
		sagittalInfo[0] = bar_sag->value();
		sagindex[0] = sagittalInfo[0];
		desiredRegion.SetSize(sagittalSize);
		desiredRegion.SetIndex(sagindex);
		extractFilter = FilterType::New();
		extractFilter->SetExtractionRegion(desiredRegion);
		extractFilter->SetInput(in);
		extractFilter->SetDirectionCollapseToIdentity();
		extractFilter->Update();

		sagImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());

		rgbextractFilter = rgbFilterType::New();
		rgbextractFilter = rgbFilterType::New();
		rgbextractFilter->SetExtractionRegion(desiredRegion);
		rgbextractFilter->SetInput(masks);
		rgbextractFilter->SetDirectionCollapseToIdentity();
		rgbextractFilter->Update();
		maskImgsag = itk::OpenCVImageBridge::ITKImageToCVMat<RGBImageType>(rgbextractFilter->GetOutput());


		cvtColor(sagImg, sagImg, COLOR_GRAY2BGR);
		cvtColor(maskImgsag, maskImgsag, COLOR_RGB2BGR);


		addWeighted(sagImg, 0.5, maskImgsag, 0.5, 0.0, sagImg);

		for (int i = 0; i < landmarks_sag.size();i++) {
			
			circle(sagImg, landmarks_sag[i], 3, Scalar(0, 0, 255), 2);
			putText(sagImg, vertebraes[i], landmarks_sag[i], cv::FONT_HERSHEY_SIMPLEX,1, Scalar(255, 0, 0),2);
		}

		QImage sagittale((const uchar*)sagImg.data, sagImg.cols, sagImg.rows, static_cast<int>(sagImg.step), QImage::Format_RGB888);
		//assiale = Mat2QImage(axialImg);
		QPixmap imgsagittale = QPixmap::fromImage(sagittale, Qt::ColorOnly);
		//imshow("Assiale1", axialImg);
		label_sag->setPixmap(imgsagittale.scaled(label_sag->width(), label_sag->height(), Qt::KeepAspectRatio));


		//CORONALE
		coronalInfo[0] = bar_cor->value();
		corindex[1] = coronalInfo[0];
		desiredRegion.SetSize(coronalSize);
		desiredRegion.SetIndex(corindex);
		extractFilter = FilterType::New();
		extractFilter->SetExtractionRegion(desiredRegion);
		extractFilter->SetInput(in);
		extractFilter->SetDirectionCollapseToIdentity();
		extractFilter->Update();

		corImg = itk::OpenCVImageBridge::ITKImageToCVMat<ImageTypeUC>(extractFilter->GetOutput());


		rgbextractFilter = rgbFilterType::New();
		rgbextractFilter = rgbFilterType::New();
		rgbextractFilter->SetExtractionRegion(desiredRegion);
		rgbextractFilter->SetInput(masks);
		rgbextractFilter->SetDirectionCollapseToIdentity();
		rgbextractFilter->Update();
		maskImgcor = itk::OpenCVImageBridge::ITKImageToCVMat<RGBImageType>(rgbextractFilter->GetOutput());




		cvtColor(corImg, corImg, COLOR_GRAY2BGR);
		cvtColor(maskImgcor, maskImgcor, COLOR_RGB2BGR);

		
		addWeighted(corImg, 0.5, maskImgcor, 0.5, 0.0, corImg);
		for (int i = 0; i < landmarks_cor.size();i++) {

			circle(corImg, landmarks_cor[i], 3, Scalar(0, 0, 255), 2);
			putText(corImg, vertebraes[i], landmarks_cor[i], cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
		}

		QImage coronale((const uchar*)corImg.data, corImg.cols, corImg.rows, static_cast<int>(corImg.step), QImage::Format_RGB888);
		//assiale = Mat2QImage(axialImg);
		QPixmap imgcoronale = QPixmap::fromImage(coronale, Qt::ColorOnly);
		//imshow("Assiale1", axialImg);
		label_cor->setPixmap(imgcoronale.scaled(label_cor->width(), label_cor->height(), Qt::KeepAspectRatio));



	} while (waitKey(100)!= ' ' &&ui.windowing_check->isChecked() == false && close_window== false);
}

void QtGuiApplication1::fromLabeltoName(std::vector<uchar>& labels, vector<cv::String>& vertebraes)
{
	for (int i = 0; i < labels.size();i++)
	{
		switch (static_cast<int>(labels[i])) 
		{
		case 0:
			vertebraes.push_back(cv::String("C1"));
		case 1:
			vertebraes.push_back(cv::String ("C2"));
		case 2:
			vertebraes.push_back(cv::String("C3"));
		case 3:
			vertebraes.push_back(cv::String("C4"));
		case 4:
			vertebraes.push_back(cv::String("C5"));
		case 5:
			vertebraes.push_back(cv::String("C6"));
		case 6:
			vertebraes.push_back( cv::String( "C7"));
		case 7:
			vertebraes.push_back(cv::String("T1"));
		case 8:
			vertebraes.push_back(cv::String("T2"));
		case 9:
			vertebraes.push_back(cv::String("T3"));
		case 10:
			vertebraes.push_back(cv::String("T4"));
		case 11:
			vertebraes.push_back(cv::String("T5"));
		case 12:
			vertebraes.push_back(cv::String("T6"));
		case 13:
			vertebraes.push_back(cv::String("T7"));
		case 14:
			vertebraes.push_back(cv::String ("T8"));
		case 15:
			vertebraes.push_back(cv::String("T9"));
		case 16:
			vertebraes.push_back(cv::String("T10"));
		case 17:
			vertebraes.push_back(cv::String("T11"));
		case 18:
			vertebraes.push_back(cv::String("T12"));
		case 19:
			vertebraes.push_back(cv::String("L1"));
		case 20:
			vertebraes.push_back(cv::String("L2"));
		case 21:
			vertebraes.push_back(cv::String("L3"));
		case 22:
			
			vertebraes.push_back(cv::String("L4"));
		case 23:
			vertebraes.push_back(cv::String("L5"));
		case 24:
			vertebraes.push_back(cv::String("L6"));
		case 25:
			vertebraes.push_back(cv::String("T13"));

		}
	}

}