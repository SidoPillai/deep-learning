import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * The class data_load is used to load training and testing images for executing Convolutional neural network.
 * 
 * @author suhaspillai
 * @version 20-Sep-2015	
 */

public class data_load {

	numjava objXTrain;
	numjava objXTest;
	numjava objYTrain;
	numjava objYTest;

	/*
	 * Constructor to initialize data members of numjava class.(i.e matrices). 
	 */
	public data_load(int trainSize,int testSize)
		{
		objXTrain = new numjava (trainSize,3072);
		objXTest  = new numjava (testSize,3072);
		objYTrain = new numjava (trainSize,1);
		objYTest = new numjava (testSize,1);
		}

	public data_load(int dataSize)
		{
		objXTrain = new numjava (dataSize,3072);
		objYTrain = new numjava (dataSize,1);
		}
	/**
	 * The method is used to load data in batches.
	 * @param filepath Path where the binary image files are stored.
	 * @param dataSize Number of images in binary image file.
	 * @return return data_load object that include matrices of numjava  
	 * @throws IOException
	 */

	public static data_load load_cifar_batch(String filepath, int dataSize) throws IOException
		{
		data_load d =new data_load(dataSize);
		File f = new File(filepath);
		FileInputStream in =new FileInputStream(f);
		int fileLength = (int)f.length();
		byte [] buffer = new byte[fileLength];
		int classLabel=0;
		int imageRowCount = 0;
		int imageColCount= 0; 
		while (in.read(buffer)!=-1);
		byte var;
		int unsignedInteger; 
		for (int i=0;i<3073000;i++)
			{
			var = buffer [i];
			unsignedInteger = var & 0xFF;
			if (i%3073==0 || i==0)
				{
				d.objYTrain.finalmatrix[classLabel][0]= (float)unsignedInteger;
				classLabel++;
				}
			else
				{
				if (imageColCount<3072)
					{
					d.objXTrain.finalmatrix[imageRowCount][imageColCount] = (float)unsignedInteger;
					imageColCount++;
					}
				else
					{
					imageRowCount++;
					imageColCount=0;
					d.objXTrain.finalmatrix[imageRowCount][imageColCount] = (float)unsignedInteger;
					imageColCount++;
					}
				}

			}
		return d;
		}

	/**
	 * The method is used to load CIFAR-10 images.
	 * @param path The path where the data is located (i.e Binary Image files.) 
	 * @param numTrain Number of images for training convolutional neural network.
	 * @param numTest  Number of images used for testing Convolutional neural network.
	 * @return Object of data_load that contains data members of numjava class(i.e matrices) 
	 * @throws IOException
	 */
	public static data_load load_CIFAR(String path, int numTrain, int numTest) throws IOException
		{
		data_load dobj1= new data_load(numTrain,numTest);
		data_load dobj2;
		String fileName = "data_batch_";
		String filePath;

		int countRow=0;
		for (int i=1; i<2; i++)
			{
			filePath=path+fileName+(String.valueOf(i))+".bin";
			dobj2=load_cifar_batch(filePath,1000);

			for (int j=0;j<1000;j++)
				{
				dobj1.objXTrain.finalmatrix[countRow]=dobj2.objXTrain.finalmatrix[j].clone();
				dobj1.objYTrain.finalmatrix[countRow]=dobj2.objYTrain.finalmatrix[j].clone();
				countRow++;
				}
			}
		countRow=0;
		fileName="test_batch";
		filePath=path+fileName+".bin";
		dobj2=load_cifar_batch(filePath,1000);
		for (int j=0;j<1000;j++)
			{
			dobj1.objXTest.finalmatrix[countRow]=dobj2.objXTrain.finalmatrix[j].clone();
			dobj1.objYTest.finalmatrix[countRow]=dobj2.objYTrain.finalmatrix[j].clone();
			countRow++;
			}

		return dobj1;
		}
	}
