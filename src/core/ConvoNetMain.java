import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;
//import edu.rit.pj2.Task;


/**
 * Class ConvoNetMain is the main class that loads the images and trains the system parallel across
 * multi-cores
 * @author suhaspillai
 *
 */

public class ConvoNetMain {

	public static  void main(String args[]) throws Exception
	{


		System.out.println("Sequential run");
		layers layer = new layers();

		/********************Sanity loss check*****************************/
		ConvoNet convNet = new ConvoNet();

		//load data
		String path = "/home/stu2/s19/sbp3624/ConvolutionalNN/data/";
		data_load d = data_load.load_CIFAR(path, 50000, 10000);
		numjava x_Train = d.objXTrain;
		numjava x_Test = d.objXTest;
		numjava y_Train = d.objYTrain;
		numjava y_Test = d.objYTest;
		int num_training = 49000;
		int num_val = 1000;

		System.out.println("Train data"+x_Train.M);
		System.out.println("test data"+x_Test.M);

		ClassifierTrainer trainer = new ClassifierTrainer();
		Map<String,Object>trainedMap = null;
		numjava xTrain = new numjava (num_training,x_Train.N);
		numjava yTrain = new numjava(num_training,1);	
		numjava xVal = new numjava (num_val,x_Train.N);
		numjava yVal = new numjava(num_val,y_Train.N);		

		//For mean normalizing
		numjava meanOfAll = numjava.getMeanNormalized(xTrain);

		for(int i = 0 ;i < num_training; i++)
		{
			xTrain.finalmatrix[i] = Arrays.copyOf(x_Train.finalmatrix[i],x_Train.finalmatrix[i].length);


			for(int j =0; j<xTrain.N;j++)
			{
				xTrain.finalmatrix[i][j] = xTrain.finalmatrix[i][j] - meanOfAll.finalmatrix[0][j]; 
			}

			yTrain.finalmatrix[i][0]=y_Train.finalmatrix[i][0];
		}

		// loading  validation data
		int counter=0;	
		for(int i = 49000 ;i < 50000 ; i++)
		{
			xVal.finalmatrix[counter] = Arrays.copyOf(x_Train.finalmatrix[i],x_Train.finalmatrix[i].length);

			for(int j =0; j < xVal.N;j++)
			{
				xVal.finalmatrix[counter][j] = xVal.finalmatrix[counter][j] - meanOfAll.finalmatrix[0][j]; 
			}

			yVal.finalmatrix[counter] = Arrays.copyOf(y_Train.finalmatrix[i], y_Train.finalmatrix[i].length);
			counter++;	
		}

		x_Train =null; // free memory 
		y_Train = null; // free memory


		double weightScale=1e-3;
		double biasScale=0;
		int rand;
		int noFilters = 32;
		int num_classes =10;
		int channels =3;
		int height =32;
		int width =32;
		int filterSize = 5;

		Random r  = new Random();
		//parameter for convolution filters.
		Map<String,Integer> convParam = new HashMap<String, Integer>();
		convParam.put("pad", (filterSize-1)/2);
		convParam.put("imageWidth",width);
		convParam.put("imageHeight",height);
		convParam.put("stride", 1);
		convParam.put("filterHeight",filterSize);
		convParam.put("filterWidth",filterSize);
		convParam.put("channels",3);

		Map<String,Object> model=convNet.layerConvonet_loadParameters(weightScale, biasScale, num_classes, noFilters, filterSize, channels, width, height);
		double reg = 0.000042;
		double lr =0.002848 ;
		Map<String,Object> output;// = trainer.train(xTrain,yTrain,xVal,yVal,model,reg,learningRate,0.9,true,1,200,convParam); 	 
		reg=0;
		Random random = new Random();
		double momentum = 0.9;
		output =trainer.train(xTrain,yTrain,xVal,yVal,model,reg,lr,0.9,true,5,200,convParam);
		double valacc= (double)output.get("bestvalidationacc");

		// Randomly choosing learning rate and regularization values.
		/*
		 for(int i = 0 ; i < 3; i++)

		{

			System.out.println("Inside lr and reg for loop");
			 lr = Math.pow(10, numjava.generateRandomRange(-4, -5, random));
			 reg = Math.pow(10,numjava.generateRandomRange(-4, 2, random));
			 model =convNet.layerConvonet_loadParameters(weightScale, biasScale, num_classes, noFilters, filterSize, channels, width, height);		
		System.out.println("Number of Training examples: "+xTrain.M);
		 output =trainer.train(xTrain,yTrain,xVal,yVal,model,reg,lr,0.9,true,5,200,convParam);

		if(valacc<(double)output.get("bestvalidationacc"))
		{
			valacc=(double)output.get("bestvalidationacc");
			System.out.println("***************************************"+lr+"        "+"       "+reg+"      "+valacc);
		}
		//long end = System.currentTimeMillis();
		//System.out.println("Total Time "+String.valueOf(end-start));	
		}
		 */

		System.out.println("best validation accuracy"+valacc);	
		System.out.println("executing");

		// Test cases to check all forward and backward propagation values on a small dataset.
		/*
Test t = new Test();
t.test();		
		 */	

	}


}
