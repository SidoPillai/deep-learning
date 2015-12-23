
import java.util.*;

/**
 * Class contains all the function used to train and manipulate images of convolutional neural network.
 * @author suhaspillai/siddeshpillai
 *
 */
class numjava
{
	public int M;
	public int N;
	public double finalmatrix [][];
	public static Random rand = new Random();
	public numjava(int M,int N)
	{
		this.M=M;
		this.N=N;
		finalmatrix=new double[M][N];
	}

	// To create random 2-d array
	public static numjava createrandom (int M,int N)
	{
		numjava mat=new numjava(M,N);
		for (int i=0;i<M;i++)
		{
			for (int j=0;j<N;j++)
			{
				mat.finalmatrix[i][j]= rand.nextDouble()*(1+1)-1;

			}
		}

		return mat; 
	}
	// To print a matrix 
	public static void print(numjava mat)
	{
		System.out.println(mat.M+" "+mat.N);
		for (int i=0;i<mat.M;i++)
		{
			for (int j=0;j<mat.N;j++)
			{
				//	System.out.printf("%9.4f ", mat.finalmatrix[i][j]);
				System.out.printf("%.10g\t",mat.finalmatrix[i][j]);
			}
			System.out.println();    
		}    

	}

	//Dot product of 2 matrices
	public static numjava dot(numjava mat1,numjava mat2)
	{

		int M1=mat1.M;
		int N1=mat1.N;
		int M2=mat2.M;
		int N2=mat2.N;

		if (N1!=M2) throw new RuntimeException("Illegal matrix dimensions");

		numjava mat=new numjava(M1,N2);

		for (int i=0;i<M1;i++)
		{
			for (int j=0;j<N2;j++)
			{   
				mat.finalmatrix[i][j]=0;
				for (int k=0;k<M2;k++)
				{
					mat.finalmatrix[i][j]+=mat1.finalmatrix[i][k]*mat2.finalmatrix[k][j];
				}
			}
		}            
		return mat;
	}

	//Element wise multiplication
	public static numjava elementmul(numjava mat1,numjava mat2)
	{
		int M1=mat1.M;
		int N1=mat1.N;
		int M2=mat2.M;
		int N2=mat2.N;
		try
		{
			if (N1!=N2 || M1!=M2)
				throw new Exception();
		}

		catch(Exception e)
		{
			System.out.println("Illegal matrix dimension");
		}

		numjava mat=null;
		if (M1!=M2)
		{   
			mat=new numjava(M2,N1);     
			for (int i=0;i<M2;i++)
			{

				for(int j=0;j<N1;j++)
				{
					mat.finalmatrix[i][j]=mat1.finalmatrix[i][0]*mat2.finalmatrix[0][j];
				}
			}
		}

		else
		{
			mat=new numjava(M1,N1);
			for (int i=0;i<M1;i++)
			{
				for (int j=0;j<N1;j++)
				{
					mat.finalmatrix[i][j]=mat1.finalmatrix[i][j]*mat2.finalmatrix[i][j];
				}
			}

		}
		return mat;    
	} 

	/*
	 * Multiply all entire matrix with one value
	 */

	public static numjava MulbyVal(numjava mat, double val)
	{
		numjava mulMat = new numjava(mat.M,mat.N);
		for(int i=0; i < mat.M ; i++)
		{
			for(int j =0; j < mat.N; j++)
			{
				mulMat.finalmatrix[i][j] = mat.finalmatrix[i][j] * val;
			}
		}
		return mulMat;
	}



	//Reshaping the  matrices
	public static numjava reshape(numjava mat,int M,int N)
	{

		int old_M=mat.M;
		int old_N=mat.N;
		int counteri=0;
		int counterj=0;
		numjava C=new numjava(M,N);
		//try
		//{
		if((old_M*old_N)!=(M*N))
		{

			throw new RuntimeException("Cannot reshape");

		}
		else 
		{

			for (int i=0;i<M;i++)
			{
				for (int j=0;j<N;j++)
				{
					if (counterj>=old_N)
					{
						counteri++;
						counterj=0;
					}

					C.finalmatrix[i][j]=mat.finalmatrix[counteri][counterj];
					counterj++;
				}

			}            

		}
		return C;       

	}

	// Creating a matrix
	public static numjava creatematrix(double mat[][])
	{   
		int M=mat.length;
		int N=mat[0].length;
		System.out.println(M+" "+N);
		numjava C=new numjava(M,N);
		for (int i=0;i<M;i++)
		{
			for(int j=0;j<N;j++)
			{
				C.finalmatrix[i][j]=mat[i][j];
				System.out.println(C.finalmatrix[i][j]);
			}
		}
		return C;
	}

	//Transposing a matrix
	public static numjava transpose(numjava mat)
	{

		numjava C=new numjava(mat.N,mat.M);

		for (int i=0;i<mat.M;i++)
		{
			for (int j=0;j<mat.N;j++)
			{

				C.finalmatrix[j][i]= mat.finalmatrix[i][j];

			}

		}
		return C;
	}

	// get max element of the entire matrix.
	static double get_Max_Element_Matrix_by_Value(numjava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		double max = matrix.finalmatrix[0][0];

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				if(matrix.finalmatrix[row][column] > max) {
					max = matrix.finalmatrix[row][column];
				}
			}
		}
		return max;
	}

	// Returns an array with position of max value in every row.
	static double[] get_Arg_Max(numjava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		double max;

		double[] index_Max = new double[matrix.M];

		for (int row = 0; row < matrix.M; row++) {
			max = matrix.finalmatrix[row][0];
			index_Max[row] = 0;
			for (int column = 1; column < matrix.N; column++) {

				if(matrix.finalmatrix[row][column] > max) {
					max = matrix.finalmatrix[row][column];
					index_Max[row] = column; 
				}
			}
		}
		return index_Max;
	}

	// Get max element row wise.
	static double get_Max_Element_Matrix_by_Row(numjava matrix, int row_Number) {

		double max = matrix.finalmatrix[row_Number][0];

		for (int row = 0; row < matrix.M; row++) {
			if(matrix.finalmatrix[row_Number][row] > max) {
				max = matrix.finalmatrix[row_Number][row];
			}
		}
		return max;
	}
	//get mean value.
	public static double getMean(double [] mat, numjava mat1)
	{
		int count=0;
		int totalsamples = mat.length;
		for (int i = 0; i < totalsamples; i++)
		{
			if (mat[i]==mat1.finalmatrix[i][0])
			{
				count++;
			}
		}

		return ((double)count/(double)totalsamples);
	}

	//get mean value
	static double getMean( double [] mat, double [] mat1)
	{

		int count=0;
		int totalsamples = mat.length;
		for (int i = 0; i < totalsamples; i++)
		{
			if (mat[i]==mat1[i])
			{
				count++;
			}
		}

		return ((double)count/(double)totalsamples);
	}


	// get max element by column
	static double get_Max_Element_Matrix_by_Column(numjava matrix, int row_Number) {

		double max = matrix.finalmatrix[row_Number][0];

		for (int column = 1; column < matrix.N; column++){
			if(matrix.finalmatrix[row_Number][column] > max) {
				max = matrix.finalmatrix[row_Number][column];
			}
		}
		return max;
	}

	// get min element by value.
	static double get_Min_Element_Matrix_by_Value(numjava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		double min = matrix.finalmatrix[0][0];

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				if(matrix.finalmatrix[row][column] < min) {
					min = matrix.finalmatrix[row][column];
				}
			}
		}
		return min;
	}

	// get min element by index
	static int[] get_Min_Element_Matrix_by_Index(numjava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		double min = matrix.finalmatrix[0][0];

		int[] index_Min = new int[2];

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				if(matrix.finalmatrix[row][column] < min) {
					min = matrix.finalmatrix[row][column];
					index_Min[0] = row;
					index_Min[1] = column;
				}
			}
		}
		return index_Min;
	}
	// get min element of matrix by row.
	static double get_Min_Element_Matrix_by_Row(numjava matrix, int row_Number) {

		double min = matrix.finalmatrix[row_Number][0];

		for (int row = 0; row < matrix.M; row++) {
			if(matrix.finalmatrix[row_Number][row] > min) {
				min = matrix.finalmatrix[row_Number][row];
			}
		}
		return min;
	}

	// get min element of matrix by column
	static double get_Min_Element_Matrix_by_Column(numjava matrix, int column_Number) {

		double min = matrix.finalmatrix[0][column_Number];

		for (int column = 0; column < matrix.N; column++){
			if(matrix.finalmatrix[column][column_Number] > min) {
				min = matrix.finalmatrix[column][column_Number];
			}
		}
		return min;
	}

	/**
	 * Calculates exponential values for all the elements in the matrix.
	 * @param mat
	 * @return matrix with exponential values
	 */

	public static numjava calculate_Exponential(numjava mat) {

		for(int i = 0; i<mat.M;i++)
		{
			for (int j = 0; j< mat.N; j++)
			{
				mat.finalmatrix[i][j] = (double)Math.exp(mat.finalmatrix[i][j]);
			}
		}

		return mat;
	}

	/**
	 * Makes a deep copy of the matrix
	 * @param mat
	 * @return
	 */

	public static numjava deepCopy (numjava mat)
	{
		numjava mat1 = new numjava(mat.M,mat.N);

		for (int i = 0;i< mat.M; i++)
		{
			mat1.finalmatrix[i] = Arrays.copyOf(mat.finalmatrix[i], mat.finalmatrix[i].length);
		}
		return mat1;
	}

	/**
	 * Adding two matrices
	 * @param mat1 
	 * @param mat2
	 * @return result after adding two matricies.
	 */
	public static numjava add(numjava mat1, numjava mat2)
	{
		// check for matrix or vector
		numjava mat = null;
		if (mat2.M==1)
		{
			try{

				if (mat1.N == mat2.N)
				{
					mat = new numjava(mat1.M,mat1.N);
					{
						for (int i=0;i<mat1.M;i++)
							for (int j=0;j<mat1.N;j++)
							{
								mat.finalmatrix[i][j]= mat1.finalmatrix[i][j] + mat2.finalmatrix[0][j];
							}
					}
				}
			}
			catch (Exception e)
			{

				System.out.println("Number of columns should match");
			}
		}

		else if (mat2.N==1)
		{

			try
			{

				if (mat1.M==mat2.M)
				{	
					mat = new numjava (mat1.N,mat1.N);
					for (int j=0;j<mat1.N;j++)
					{
						for (int i=0; i<mat1.M;i++)
						{
							mat.finalmatrix[i][j] = mat1.finalmatrix[i][j] + mat2.finalmatrix[i][0];
						}
					}
				}
			}
			catch (Exception e)
			{

				System.out.println("Numbers of rows should match");
			}

		}

		else 
		{

			try
			{
				if (mat1.M==mat2.M && mat1.N==mat2.N)
				{	
					mat = new numjava (mat1.M,mat1.N);
					for (int i=0;i<mat1.M;i++)
					{
						for (int j=0;j<mat1.N;j++)
						{
							mat.finalmatrix[i][j] = mat1.finalmatrix[i][j] + mat2.finalmatrix[i][j];
						}
					}
				}
			}
			catch (Exception e)
			{

				System.out.println("Numbers of rows and colums should match");
			}
		}
		return mat;
	}

	// adding matrix and a vector
	public static numjava addByRow(numjava mat, numjava mat1)
	{
		int iter = mat.M;
		for(int i = 0;i<iter;i++)
		{
			mat.finalmatrix[i] = addByVal(mat.finalmatrix[i],mat1.finalmatrix[0][i]);
		}
		return mat;
	}

	// add every element of the matrix with a single value
	public static double[] addByVal(double [] mat, double val)
	{
		for (int i = 0; i<mat.length; i++)
		{
			mat[i] = mat[i]+val;
		}
		return mat;
	}

	/**
	 * Subtract val from all the elements of the array.
	 * @param x matrix 
	 * @param val value to be subtracted
	 * @return matrix with subtracted value.
	 */
	public static double [] sub(double [] x, double val)
	{
		for(int i = 0 ; i<x.length; i++)
		{
			x[i] = x[i] - val;
		}
		return x;
	}

	/**
	 * Divide a matrix by another matrix
	 */
	public static numjava divide (numjava mat1, numjava mat2)
	{
		if (mat2.N==1) // mat2 is a vector
		{
			for(int i = 0 ;i<mat1.M; i++)
			{
				for (int j = 0 ; j<mat1.N; j++)
				{
					// Basically, it is every row of mat1 is getting divide by a value from mat 2 , which is single dimension
					mat1.finalmatrix[i][j]=mat1.finalmatrix[i][j]/mat2.finalmatrix[i][0];

				}

			}
		}
		else if (mat1.N==mat2.N)
		{
			for (int i = 0; i<mat1.M;i++)
			{
				for(int j = 0; j < mat1.N; j++)
				{
					mat1.finalmatrix[i][j] = mat1.finalmatrix[i][j]/mat2.finalmatrix[i][j];

				}
			}
		}
		return mat1;
	}

	/**
	 * Subtract a matrix from another matrix
	 */

	public static numjava sub(numjava mat1, numjava mat2)
	{
		//check dimension before subtracting
		numjava mat = null;
		if(mat1.M==mat2.M && mat1.N==mat2.N)
		{
			mat = new numjava(mat1.M,mat1.N);

			for (int i = 0; i< mat1.M; i++)
			{
				for(int j = 0; j< mat1.N; j++)
				{
					mat.finalmatrix[i][j] = mat1.finalmatrix[i][j] - mat2.finalmatrix[i][j]; 
				}
			}
		}
		return mat;


	}

	/*
	 * Subtract all the elements of the matrix by a single value
	 */

	public static double[] subByVal(double[] mat1, double val)
	{
		double matsub []= new double[mat1.length];
		for(int i = 0;i<mat1.length;i++)
		{
			matsub[i] = mat1[i] - val; 
		}
		return matsub;
	}


	/**
	 * Divide a matrix by element
	 * @param mat1
	 * @param val
	 * @return result after division
	 */
	public static numjava divideByVal (numjava mat1, double val)
	{
		for(int i = 0 ;i<mat1.M; i++)
		{
			for (int j = 0 ; j<mat1.N; j++)
			{
				// Basically, it is every row of mat1 is getting divide by a value from mat 2 , which is single dimension
				mat1.finalmatrix[i][j]=mat1.finalmatrix[i][j]/val;
			}

		}
		return mat1;
	}

	/*
	 * To get sum of elements along row or column
	 * axis = 0 : sum along row 
	 * axis = 1 : sum along column
	 */
	public static numjava sum(numjava mat1, int axis)
	{
		numjava mat = null;
		double sum = 0;
		try 
		{

			if (mat1.finalmatrix!=null)
			{
				if (axis == 0)
				{

					// sum along row axis
					mat = new numjava(1,mat1.N);
					for (int i = 0 ;i<mat1.N ; i++)
					{	sum = 0;
					for (int j = 0; j<mat1.M; j++)
					{
						sum = sum + mat1.finalmatrix[j][i];	
					}
					mat.finalmatrix[0][i] = sum;
					}

				}
				else if (axis == 1) 
				{
					// add along column axis

					mat = new numjava(mat1.M,1);
					for (int i = 0; i< mat1.M; i++)
					{
						sum = 0;
						for (int j = 0; j<mat1.N; j++)
						{
							sum = sum + mat1.finalmatrix[i][j];
						}
						mat.finalmatrix[i][0] = sum;
					}

				}

			}
		}
		catch (NullPointerException e)
		{
			System.out.println("Null pointer exception");
		}

		return mat;
	}




	/**
	 * The method converts image to columns of matrix , where each  row size is equal to filter width * filter height * depth of the image.
	 * @param mat1 matrix with every image converted into rows of size equal to filter width * filter height * depth of the image and columns same as that of original image.
	 * @param mat original image 
	 * @param odimWidth output image width
	 * @param odimHeight output image height
	 * @param channels number of channels
	 * @param filterSize filter size of an image
	 * @param noOfFilters numbers of filters of  the image
	 * @param imindex index of a particular image
	 * @param stride stride to be taken
	 * @param pad padding of the image
	 * @param widthBefPad with before padding
	 * @param heigthBefPad height before padding
	 * @return mat1.
	 */

	public static numjava im2col(numjava mat1, numjava mat, int odimWidth,int odimHeight,int channels, int filterSize, int noOfFilters, int imindex, int stride, int pad, int widthBefPad,int heigthBefPad )
	{

		numjava matim2col = mat1;
		int col = 0;
		int rowInc = 0;
		boolean flag = true;
		int imHeight = odimHeight;
		int imWidth = odimWidth;
		int channelCount=0;
		int iterCount = 0;
		int j;
		int mul = 1;
		int imrowCount=0;
		int temp;
		int totalIter = filterSize * filterSize;
		int pixelsToMove = (widthBefPad*heigthBefPad)+(widthBefPad*pad*2)+(heigthBefPad*pad*2)+((pad*pad)*4); // To move to another channel(Total 32*32 + padding pixels) 
		//int pixelsToMove = 
		for (int rowCount = 0; rowCount<imHeight; rowCount++)
		{
			if(flag == true)
			{
				rowInc = 0;
				flag = false;
			}
			else
			{
				rowInc = rowInc  + (widthBefPad+2*pad)*stride; // Traverse or take stride vertically and continue with column stride,because now the image is just a single row of pixels RGB.
			}
			temp = rowInc; 
			for (int colCount = 0; colCount<imWidth; colCount++)
			{
				channelCount = 0;
				iterCount = 0;
				//i = rowCount;
				j = rowInc ;
				mul = 1;
				imrowCount = 0;
				while (channelCount<channels)
				{

					while(iterCount<totalIter)
					{

						matim2col.finalmatrix[imrowCount][col] = mat.finalmatrix[imindex][j]; // Filling value to to get im2col

						if((imrowCount+1)%filterSize==0) //imrowCount will be  multiple of the filters width.
						{
							j = j + ((widthBefPad+2*pad)-filterSize) + 1;
						}
						else
						{
							j++;  // increment every pixel value
						}
						imrowCount++;   // For 3*3 filter this will go till 9 For R, 9-18 for G, 18-27 G.
						iterCount++;
					}
					iterCount = 0;

					j = rowInc + (pixelsToMove*mul); // Go from R(0-1023) to G (1024-2047) to B (2048-3071)
					mul++; // 1 for R, 2 for G
					channelCount++; // Increment the channel, go to next channel
				}
				rowInc = rowInc + stride; // Take stride to the right 
				imrowCount = 0; // New value to be calculated for the next column so initialize back to zero 
				col++; // Increment the column 


			}
			rowInc = temp; // initialize it back to the original value, for  vertical stride.

		}

		return matim2col;
	}

	/**
	 * To generate a padded image.
	 * @param mat image that needs to be padded
	 * @param pad padding value
	 * @param val  padding value
	 * @param channels number of channels
	 * @param imWidth image width
	 * @param imHeight image height
	 * @return padded image
	 */
	public static numjava pad (double [] mat, int pad , int val, int channels, int imWidth, int imHeight)
	{
		int N = ((imWidth*imHeight)+(imWidth*pad*2)+(imHeight*pad*2)+((pad*pad)*4))*channels; //(4 for 4 corners) (Total pixels after padding)
		numjava matpad = new numjava (1,N);
		int channelCount = 0;
		int count=0;
		int padval = val;
		int j=0;
		int orgcount=0;
		while (channelCount<channels)
		{
			for (int i=0;i<(imHeight+pad*2);i++)
			{
				if(i<pad || i>=((imHeight+pad)))    // for first and last row padding 
				{
					while (count<(imWidth+pad*2))   // Do it for all columns 
					{
						matpad.finalmatrix[0][j] = padval;
						count++;
						j++;
					}
					count=0;
				}
				else
				{
					while(count<pad)      // for first/second...pad cols
					{	
						matpad.finalmatrix[0][j] = padval;
						count++;
						j++;
					}
					count=0;
					while(count<imWidth)    // Rest of the cols value of the old matrix
					{
						matpad.finalmatrix[0][j] = mat[orgcount];
						count++;
						j++;
						orgcount++;
					}
					count=0;
					while(count<pad)     // again for last cols padding.
					{

						matpad.finalmatrix[0][j] = padval;
						count++;
						j++;
					}

					count=0;
				}
			}
			channelCount++; // For the next channel.

		}
		mat=null;
		return matpad;

	}

	/**
	 * To get the original image before deconvolution.
	 * @param originalmat original image dimension before performing convolution
	 * @param mat mat from which deconvolve matrix is to be obtained.
	 * @param p padding value
	 * @param filterWidth filter width of the image
	 * @param channels number of channels
	 * @param stride stride value
	 * @param odimWidth output dimension width
	 * @param imHeightVal original image height
	 * @param imWidthVal original image width 
	 * @return deconvolved matrix.
	 */
	public static double[] deConvolve (double[] originalmat,numjava mat, int p, int filterWidth, int channels, int stride, int odimWidth, int imHeightVal, int imWidthVal)
	{
		int channelCount = 0;
		int imHeight =  imHeightVal; // original image height
		int imWidth = imWidthVal;    // original image width
		int pad = p;
		int filterW = filterWidth;
		int row = mat.M;   // size of row and col of im2col matrix
		int col = mat.N;
		int iterFilter = filterW*filterW;    // like for 3*3 iterFilter will be 9 , not considering channels
		int j;
		int count=0; 
		int rowInc = 0;   // Helps to move from one channels to another and also used for horizontal and vertical traversing.
		int rCount = 0;
		int cCount = 0;
		int widthBefPad = imWidthVal;
		//Considering images of same width and height.
		int pixelsToMove = (widthBefPad*widthBefPad)+(widthBefPad*pad*2)+(widthBefPad*pad*2)+((pad*pad)*4); // To move from R-->G--B
		//int N = ((imWidth*imHeight)+(imWidth*pad*2)+(imHeight*pad*2)+((pad*pad)*4))*channels; //(4 for 4 corners) (Total pixels after padding)
		int N = (pixelsToMove)*channels;
		numjava matcol2im = new numjava (1,N);  // array with padded pixels
		for (int colCount = 0; colCount<col; colCount++)
		{

			if(colCount!=0)   // For checking whether we have traversed to the end of the image width
			{
				// If started from row zero , then once the end of the width is reached mov eto row one, then to row two ......
				if ((colCount)%(odimWidth) == 0)
				{
					cCount++; // To track the complete width traversing (i.e In a 5*5 matrix, after traversing across 5 columns move to next row )
					//rowInc = ((widthBefPad+2*pad)*stride)*cCount;  // Traverse  row wise (i.e when 5 * 5 , width 5 is covered move to next row and  start traversing)
					rowInc = ((widthBefPad+2*pad)*stride*cCount); // correct
				}
				else
				{
					rowInc=rowInc+stride; // for horizontal stride across ones you switch to a new row.
				}
			}
			j = rowInc;
			count = 0;
			rCount = 0;
			for (int rowCount = 0; rowCount<row; rowCount++)
			{
				if(rCount<iterFilter)   // Signify end of first column (i.e all the 9 values of R, B and G are added)
				{
					// For 3 by 3 filter it will go to fisrt 3 Red pixels then the next 3 and then the next 3 , row wise, Imagine a square block
					if(count<filterW)
					{
						matcol2im.finalmatrix[0][j] = matcol2im.finalmatrix[0][j]+ mat.finalmatrix[rowCount][colCount];
						j++;
						count++;
						rCount++; //change
					}
					else
					{
						j = j + ((widthBefPad+2*pad)-filterW);  //change//this is corrected.
						count = 0;
						rowCount--;

					}
				}
				else
				{
					if (channelCount<channels-1)
					{

						rCount = 0;
						channelCount++;
						j = rowInc + pixelsToMove*channelCount;
						rowCount--;
						count=0;
					}
				}
			}
			channelCount=0;

		}
		// Now get back the original image.
		// dimension of an image before any padding was appliend on it.
		int counter=0;
		j=0;
		count=0;
		channelCount=0;
		while (channelCount<channels)
		{
			for (int i=0;i<(imHeight+pad*2);i++)
			{
				// Not to put padded values.
				if(i<pad || i>=((imHeight+pad)))    // for first and last row padding 
				{
					while (count<(imWidth+pad*2))   // Do it for all columns 
					{
						count++;
						j++;
					}
					count=0;
				}
				else
				{
					// Not to put padded values.
					while(count<pad)      // for first/second...pad cols
					{	
						count++;
						j++;
					}
					count=0;
					// put only non padded values.
					while(count<imWidth)    // Rest of the cols value of the old matrix
					{
						originalmat[counter] = matcol2im.finalmatrix[0][j];
						count++;
						j++;
						counter++;
					}
					count=0;
					// Not to put padded values.
					while(count<pad)     // again for last cols padding.
					{

						count++;
						j++;
					}

					count=0;
				}
			}
			channelCount++; // For the next channel.

		}
		//	return matpad;
		return originalmat;
	}

	/**
	 * To calculate relative error between two matrices
	 * @param a matrix a 
	 * @param b matrix b
	 * @return relative error between the two matricies.
	 * @throws Exception
	 */
	public static double relError(numjava a, numjava b) throws Exception
	{
		numjava relmat = new numjava(a.M,a.N); 
		for(int i = 0;i<a.M;i++)
		{
			for (int j = 0; j< a.N; j++)
			{

				double deno = (double)Math.max(0.00000001,Math.abs(a.finalmatrix[i][j])+Math.abs(b.finalmatrix[i][j]));
				relmat.finalmatrix[i][j] = Math.abs(a.finalmatrix[i][j]-b.finalmatrix[i][j])/deno;
			}
		}
		double max  = get_Max_Element_Matrix_by_Value(relmat);
		relmat = null;
		return max;

	}

	/**
	 * Gets an array with number of elements equal to batch size.
	 * @param N Total sample size
	 * @param batchSize Total bacth size
	 * @param x numjava object containing number of training samples.
	 * @param randMaskArray store random imgaes from 0 - total samples 
	 */

	public static void randomMask(int N, int batchSize, numjava x, numjava randMaskArray)
	{
		int rand;
		Random r  = new Random();
		for (int i = 0; i< batchSize; i++)
		{
			rand = r.nextInt(N-0);
			System.out.println("random no of training: "+rand);
			randMaskArray.finalmatrix[i] = x.finalmatrix[rand]; 
		}

	}

	/**
	 * To get random numbers between 0 to total number of images
	 * @param N Total sample size
	 * @param batchSize Total batch size
	 * @param x numjava object containing number of training samples.
	 * @param randArray Array which store the random points. 
	 */
	public static void getrandom(int end, int batchSize, int randArray [])
	{
		int rand;
		Random r  = new Random();
		for (int i = 0; i< batchSize; i++)
		{
			rand = r.nextInt(end-0);
			randArray[i]=rand; 
		}

	}

	// get equally space numbers in the given range
	public static  numjava linspace(double start, double end, int inputSize)
	{
		numjava mat = new numjava(1,inputSize);
		double interval = (end-start)/(inputSize-1);
		int i = 0;

		while((start-end)<0)
		{
			mat.finalmatrix[0][i] = start;

			start = start + interval; 

			i++;

		}
		// Sometimes issue while start and end are almost same, so beeter to put last value as end, else it just prints 0
		if(i<inputSize)
		{
			mat.finalmatrix[0][i]=end;
		}
		return mat;
	}


	public static void getRandomInt(numjava x, int[] copyArray)
	{

		for (int i=0;i<copyArray.length;i++)
		{
			x.finalmatrix[i][0] = copyArray[i]; 
		}

	}

	//get mean normalized matrix
	public static numjava getMeanNormalized(numjava xTrain)
	{
		numjava mean = sum(xTrain, 0);
		numjava.divideByVal(mean, xTrain.M);
		return mean;
	}


	public static double generateRandomRange(int start, int end, Random random)
	{
		double fraction =0;
		double num=0;
		if(end>start)
		{
			double range = start - end;
			fraction = range*random.nextDouble();
			num  = start + fraction;
		}
		return num;
	}

}	


