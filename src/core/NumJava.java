package core;

import java.util.*;

public class NumJava {
	int M;
	int N;

	public float finalmatrix [][];

	public NumJava(int M,int N)	{
		this.M = M;
		this.N = N;
		finalmatrix = new float[M][N];
	}

	// tested
	// To create random 2-d array
	public static NumJava createrandom (int M,int N){
		NumJava mat = new NumJava(M,N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				mat.finalmatrix[i][j] = (float)Math.random();
			}
		}
		return mat; 
	}

	// tested
	// To print a matrix 
	public static void print(NumJava mat) {
		System.out.println( mat.M + " " + mat.N);
		for (int i = 0; i < mat.M; i++) {
			for (int j = 0;j < mat.N; j++) {
				System.out.printf("%9.4f ", mat.finalmatrix[i][j]);
			}
			System.out.println();    
		}    
	}

	// tested
	//Dot product of 2 matrices
	public static NumJava dot(NumJava mat1, NumJava mat2) {

		int M1 = mat1.M;
		int N1 = mat1.N;
		int M2 = mat2.M;
		int N2 = mat2.N;

		if (N1 != M2) throw new RuntimeException("Illegal matrix dimensions");

		NumJava mat = new NumJava(M1,N2);

		for (int i = 0; i < M1; i++) {
			for (int j = 0; j < N1; j++) {   

				mat.finalmatrix[i][j] = 0;
				for (int k = 0; k < M2; k++) {
					mat.finalmatrix[i][j] += mat1.finalmatrix[i][k] * mat2.finalmatrix[k][j];
				}
			}
		}            
		return mat;
	}

	// tested
	//Elementwise multiplication
	public static NumJava elementMul(NumJava mat1, NumJava mat2) {
		int M1 = mat1.M;
		int N1 = mat1.N;
		int M2 = mat2.M;
		int N2 = mat2.N;

		try {
			if (N1 != M2) throw new Exception();
		} catch(Exception e) {
			System.out.println("Illegal matrix dimension");
		}

		NumJava mat = null;

		if (M1 != M2) {   

			mat = new NumJava(M2,N1);     

			for (int i = 0; i < M2;i++) {
				for(int j = 0; j < N1; j++) {
					mat.finalmatrix[i][j] = mat1.finalmatrix[i][0] * mat2.finalmatrix[0][j];
				}
			}
		} else {
			mat = new NumJava(M1,N1);
			for (int i = 0; i < M1; i++) {
				for (int j = 0; j < N1; j++) {
					mat.finalmatrix[i][j] = mat1.finalmatrix[i][j] * mat2.finalmatrix[i][j];
				}
			}
		}
		return mat;    
	} 

	/*
    public static void shape(float mat1[][])
    {
       return mat1[0].length,mat1[1].length;     
    }
	 */


	//Reshaping the  matrices
	public static NumJava reshape(NumJava mat,int M,int N) {

		int old_M = mat.M;
		int old_N = mat.N;
		int counteri = 0;
		int counterj = 0;

		NumJava C = new NumJava(M,N);
		//try
		//{
		if((old_M * old_N) != (M * N)) throw new RuntimeException("Cannot reshape");
		else {
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < N; j++) {
					if (counterj >= old_N) {
						counteri++;
						counterj=0;
					}

					C.finalmatrix[i][j] = mat.finalmatrix[counteri][counterj];
					counterj++;
				}
			}            
		}

		//}    
		/*
         catch(Exception e)
         {
         }
		 */       
		return C;       
		//   System.out.println("Cannot rehape matrix");

	}

	// tested
	// Creating a matrix
	public static NumJava creatematrix(float mat[][]) {   
		int M = mat.length;
		int N = mat[0].length;
		System.out.println(M + " " + N);
		NumJava C = new NumJava(M,N);

		for (int i = 0; i < M; i++) {
			for(int j = 0;j < N; j++) {
				C.finalmatrix[i][j] = mat[i][j];
				System.out.println(C.finalmatrix[i][j]);
			}
		}
		return C;
	}

	// tested
	//Transposing a matrix
	public static NumJava transpose(NumJava mat) {
		NumJava C = new NumJava(mat.N,mat.M);
		//    int counteri=0;
		//    int counterj=0;
		for (int i = 0; i < mat.M; i++) {
			for (int j = 0; j < mat.N; j++) {
				C.finalmatrix[j][i] = mat.finalmatrix[i][j];
			}
		}
		return C;
	}


	/*
static int get_Row_Size(double [][] matrix){
		return matrix.length;
	}

	static int get_Column_Size(double [][] matrix) {
		return matrix[0].length;
	}
	 */

	// tested
	static float get_Max_Element_Matrix_by_Value(NumJava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		float max = matrix.finalmatrix[0][0];

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				if(matrix.finalmatrix[row][column] > max) {
					max = matrix.finalmatrix[row][column];
				}
			}
		}
		return max;
	}

	// tested
	// Returns an array 1st index is row and 2nd is column
	static int[] get_Arg_Max(NumJava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		float max = matrix.finalmatrix[0][0];

		int[] index_Max = new int[2];

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				if(matrix.finalmatrix[row][column] > max) {
					max = matrix.finalmatrix[row][column];
					index_Max[0] = row;
					index_Max[1] = column;
				}
			}
		}
		return index_Max;
	}

	// tested
	static float get_Max_Element_Matrix_by_Row(NumJava matrix, int row_Number) {

		float max = matrix.finalmatrix[row_Number][0];

		for (int row = 0; row < matrix.M; row++) {
			if(matrix.finalmatrix[row_Number][row] > max) {
				max = matrix.finalmatrix[row_Number][row];
			}
		}
		return max;
	}

	// tested
	static float get_Max_Element_Matrix_by_Column(NumJava matrix, int column_Number) {

		float max = matrix.finalmatrix[0][column_Number];

		for (int column = 0; column < matrix.N; column++) {
			if(matrix.finalmatrix[column][column_Number] > max) {
				max = matrix.finalmatrix[column][column_Number];
			}
		}
		return max;
	}

	// tested
	static float get_Min_Element_Matrix_by_Value(NumJava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		float min = matrix.finalmatrix[0][0];

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				if(matrix.finalmatrix[row][column] < min) {
					min = matrix.finalmatrix[row][column];
				}
			}
		}
		return min;
	}

	// tested
	// Returns an array 1st index is row and 2nd is column
	static int[] get_Min_Element_Matrix_by_Index(NumJava matrix) throws Exception {

		if(matrix.finalmatrix == null) {
			throw new Exception("Matrix is null");
		}

		float min = matrix.finalmatrix[0][0];

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

	// tested
	static float get_Min_Element_Matrix_by_Row(NumJava matrix, int row_Number) {

		float min = matrix.finalmatrix[row_Number][0];

		for (int row = 0; row < matrix.M; row++) {
			if(matrix.finalmatrix[row_Number][row] > min) {
				min = matrix.finalmatrix[row_Number][row];
			}
		}
		return min;
	}

	// tested
	static float get_Min_Element_Matrix_by_Column(NumJava matrix, int column_Number) {

		float min = matrix.finalmatrix[0][column_Number];

		for (int column = 0; column < matrix.N; column++){
			if(matrix.finalmatrix[column][column_Number] > min) {
				min = matrix.finalmatrix[column][column_Number];
			}
		}
		return min;
	}

	// tested
	static float calculate_Exponential(float elem) {
		return (float)Math.exp(elem);
	}

	// tested
	// transformed with the matrix reference
	static  NumJava transform_Matrix_with_Exponential(NumJava matrix) {

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				matrix.finalmatrix[row][column] = calculate_Exponential(matrix.finalmatrix[row][column]);
			}
		}
		return matrix;
	}

	/*
	static double [][] copy_Matrix(double [][] matrix) {
		double [][] cp_Matrix = new double[matrix.length][];

		for(int i = 0; i < matrix.length; i++) {
		  double[] aMatrix = matrix[i];
		  int aLength = aMatrix.length;
		  cp_Matrix[i] = new double[aLength];
		  System.arraycopy(aMatrix, 0, cp_Matrix[i], 0, aLength);
		}
		return cp_Matrix;
	}
	 */

	// tested
	static NumJava make_Unit_Matrix(NumJava matrix) {

		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
				matrix.finalmatrix[row][column] = 1;
			}
		}
		return matrix;
	}

	// tested
	/**
	 * Adding two matrices
	 * @param mat1 
	 * @param mat2
	 * @return
	 */
	public static NumJava add(NumJava mat1, NumJava mat2) {
		// check for matrix or vector
		NumJava mat = null;
		if (mat2.M == 1) {
			try {
				if (mat1.N == mat2.N) {
					mat = new NumJava(mat1.M,mat1.N);
					for (int i=0;i<mat1.M;i++) {
						for (int j=0;j<mat1.N;j++) {
							mat.finalmatrix[i][j]= mat1.finalmatrix[i][j] + mat2.finalmatrix[0][j];
						}
					}
				}
			} catch (Exception e) {
				System.out.println("Number of columns should match");
			}
		} else if (mat2.N == 1) {
			try{
				if (mat1.M == mat2.M) {	
					mat = new NumJava (mat1.N,mat1.N);
					for (int j=0;j<mat1.N;j++) {
						for (int i=0; i<mat1.M;i++) {
							mat.finalmatrix[i][j] = mat1.finalmatrix[i][j] + mat2.finalmatrix[i][0];
						}
					}
				}
			} catch (Exception e) {
				System.out.println("Numbers of rows should match");
			}
		} else {
			try {
				if (mat1.M==mat2.M && mat1.N==mat2.N) {	
					mat = new NumJava (mat1.M,mat1.N);
					for (int i=0;i<mat1.M;i++) {
						for (int j=0;j<mat1.N;j++) {
							mat.finalmatrix[i][j] = mat1.finalmatrix[i][j] + mat2.finalmatrix[i][j];
						}
					}
				}
			} catch (Exception e) {
				System.out.println("Numbers of rows and colums should match");
			}
		}
		return mat;
	}

	// tested
	/**
	 * To get sum of elements along row or column
	 * axis = 0 : sum along row 
	 * axis = 1 : sum along column
	 */
	public static NumJava sum(NumJava mat1, int axis) {
		NumJava mat = null;
		float sum = 0;

		try {

			if (mat1.finalmatrix != null) {
				if (axis == 0) {

					// sum along row axis
					mat = new NumJava(1,mat1.N);
					for (int j = 0 ; j < mat1.M ; j++) {
						for (int i = 0; i < mat1.N; i++) {
							sum = sum + mat1.finalmatrix[i][j];	
						}
						mat.finalmatrix[0][j] = sum;
					}
				} else if (axis == 1) {
					// add along column axis

					mat = new NumJava(mat1.M,1);
					for (int i = 0; i< mat1.M; i++) {
						for (int j = 0; j<mat1.N; j++) {
							sum = sum + mat1.finalmatrix[i][j];
						}
						mat.finalmatrix[i][0] = sum;
					}
				}
			}
		} catch (NullPointerException e) {
			System.out.println("Null pointer exception");
		}
		return mat;
	}



	/**
	 * Calculates exponential values for all the elements in the matrix.
	 * @param mat
	 * @return
	 */
	// tested
	public static NumJava calculate_Exponential(NumJava mat) {

		for(int i = 0; i<mat.M;i++) {
			for (int j = 0; j< mat.N; j++) {
				mat.finalmatrix[i][j] = (float)Math.exp(mat.finalmatrix[i][j]);
			}
		}

		return mat;
	}

	/**
	 * Makes a deep copy of the matrix
	 * @param mat
	 * @return
	 */
	// tested
	public static NumJava deepCopy (NumJava mat) {
		NumJava mat1 = new NumJava(mat.M,mat.N);
		for (int i = 0;i< mat.M; i++) {
			mat1.finalmatrix[i] = Arrays.copyOf(mat.finalmatrix[i], mat.finalmatrix[i].length);
		}
		return mat1;
	}

	/**
	 * Subtract val from all the elements of the array.
	 * @param x
	 * @param val
	 * @return
	 */
	// tested
	public static float [] sub(float [] x, float val) {
		for(int i = 0 ; i<x.length; i++) {
			x[i] = x[i] - val;
		}
		return x;
	}

	// tested
	/**
	 * Divide a matrix by another matrix
	 * @param mat1
	 * @param mat2
	 * @return
	 */
	public static NumJava divide(NumJava mat1, NumJava mat2) {
		if (mat2.N == 1) { // mat2 is a vector 			
			for(int i = 0 ; i < mat1.M; i++){
				for (int j = 0 ; j < mat1.N; j++) {
					// Basically, it is every row of mat1 is getting divide by a value from mat 2 , which is single dimension
					mat1.finalmatrix[i][j] = mat1.finalmatrix[i][j] / mat2.finalmatrix[i][0];
				}
			}
		} else if (mat1.N == mat2.N) {
			for (int i = 0; i < mat1.M; i++) {
				for(int j = 0; j < mat1.N; j++) {
					mat1.finalmatrix[i][j] = mat1.finalmatrix[i][j] / mat2.finalmatrix[i][j];
				}
			}
		}
		return mat1;
	}

	//tested
	/**
	 * Subtract a matrix from another matrix
	 * @param mat1
	 * @param mat2
	 * @return
	 */
	public static NumJava sub(NumJava mat1, NumJava mat2) {
		//check dimension before subtracting
		NumJava mat = null;
		if(mat1.M == mat2.M && mat1.N == mat2.N) {
			mat = new NumJava(mat1.M,mat1.N);
			for (int i = 0; i < mat1.M; i++) {
				for(int j = 0; j < mat1.N; j++) {
					mat.finalmatrix[i][j] = mat1.finalmatrix[i][j] - mat2.finalmatrix[i][j]; 
				}
			}
		}
		return mat;
	}

	//tested
	/**
	 * Divide a matrix by element
	 * @param mat1
	 * @param val
	 * @return
	 */
	public static NumJava divideByVal(NumJava mat1, float val) {
		for(int i = 0 ;i<mat1.M; i++) {
			for (int j = 0 ; j<mat1.N; j++) {
				// Basically, it is every row of mat1 is getting divide by a value from mat 2 , which is single dimension
				mat1.finalmatrix[i][j]=mat1.finalmatrix[i][j]/val;
			}
		}
		return mat1;
	}

	// tested
	/**
	 * The functionis used to convert image to column. (im2col)
	 * channels - rgb
	 * filtersize - filter matrix dimensions
	 * stride - shift value
	 * pad - padding factor
	 * 
	 */
	public static NumJava im2col(NumJava newMatrix, NumJava mat, int newdimWidth,int newdimHeight,int channels, int filterSize, 
			int noOfFilters, int imindex, int stride, int pad, int widthBefPad,int heigthBefPad ) {
		//numjava matim2col = new numjava(noOfFilters, odimWidth*odimHeight);
		NumJava matim2col = newMatrix;
		int col = 0;
		int rowInc = 0;
		boolean flag = true;
		int imHeight = newdimHeight;
		int imWidth = newdimWidth;
		int channelCount=0;
		int iterCount = 0;
		int j;
		int mul = 1;
		int imrowCount=0;
		int temp;
		int totalIter = filterSize * filterSize;
		int pixelsToMove = (widthBefPad*heigthBefPad)+(widthBefPad*pad*2)+(heigthBefPad*pad*2)+((pad*pad)*4); // To move to another channel(Total 32*32 + padding pixels) 
		//int pixelsToMove = 
		for (int rowCount = 0; rowCount<imHeight; rowCount++) {
			if(flag == true) {
				rowInc = 0;
				flag = false;
			} else {
				rowInc = rowInc  + (widthBefPad+2*pad)*stride; // Traverse or take stride vertically and continue with column stride,because now the image is just a single row of pixels RGB.
			}
			temp = rowInc; 
			for (int colCount = 0; colCount<imWidth; colCount++) {
				channelCount = 0;
				iterCount = 0;
				//i = rowCount;
				j = rowInc ;
				mul = 1;
				imrowCount = 0;
				while (channelCount<channels) {
					while(iterCount<totalIter) {
						//						matim2col.finalmatrix[imrowCount][col] = mat.finalmatrix[imindex][j];
						//						matim2col.finalmatrix[++imrowCount][col] = mat.finalmatrix[imindex][j+1];
						//						matim2col.finalmatrix[++imrowCount][col] = mat.finalmatrix[imindex][j+2];

						matim2col.finalmatrix[imrowCount][col] = mat.finalmatrix[imindex][j]; // Filling value to to get im2col

						//j++;

						if((imrowCount+1)%filterSize==0) //imrowCount will be  multiple of the filters width.
						{
							//j=j+(imWidth-filterSize)+1; // Get the next position RGB rows
							j = j + ((widthBefPad+2*pad)-filterSize)+1;
						} else {
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

	// tested
	/**
	 * 
	 * @param mat
	 * @param pad
	 * @param val
	 * @param channels
	 * @param imWidth
	 * @param imHeight
	 * @return
	 */
	public static NumJava pad (float [] mat, int pad , int val, int channels, int imWidth, int imHeight) {
		//int M = mat.M + pad*2; // evenly padding both horizontal sides the sides of the image.
		//int N = mat.N + pad*2;// evenly padding both vertical sides of the image.
		int N = (imWidth*imHeight)+(imWidth*pad*2)+(imHeight*pad*2)+((pad*pad)*4); //(4 for 4 corners) (Total pixels after padding)
		NumJava matpad = new NumJava (1,N);
		int channelCount = 0;
		//		boolean flag = true;
		int count=0;
		int padval = val;
		//		int iter = pad*2*imWidth;
		int j=0;
		int orgcount=0;
		while (channelCount<channels) {
			for (int i=0;i<(imHeight+pad*2);i++) {
				if(i<pad || i>=((imHeight+pad)))    // for first and last row padding 
				{
					while (count<(imWidth+pad*2))   // Do it for all columns 
					{
						matpad.finalmatrix[0][j] = padval;
						count++;
						j++;
					}
					count=0;
				} else {
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
				//j++;
			}
			channelCount++; // For the next channel.
		}
		return matpad;
	}

	/**
	 * To get back original image without any padding and convolution.
	 */ 
	public static NumJava deConvolve (NumJava originalmat,NumJava mat, int p, int filterWidth, 
			int channels, int stride, int odimWidth, int imHeightVal, int imWidthVal) {
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
		NumJava matcol2im = new NumJava (1,N);  // array with padded pixels
		for (int colCount = 0; colCount<col; colCount++) {

			if(colCount!=0)   // For checking whether we have traversed to the end of the image width
			{
				// If started from row zero , then once the end of the width is reached mov eto row one, then to row two ......
				if ((colCount)%odimWidth == 0) {
					cCount++; // To track the complete width traversing (i.e In a 5*5 matrix, after traversing across 5 columns move to next row )
					rowInc = ((widthBefPad+2*pad)*stride)*cCount;  // Traverse  row wise (i.e when 5 * 5 , width 5 is covered move to next row and  start traversing)
				} else {
					rowInc=rowInc+stride; // for horizontal stride across ones you switch to a new row.
				}
			}
			j = rowInc;
			for (int rowCount = 0; rowCount<row; rowCount++) {
				
			
					if(rCount<iterFilter)   // Signify end of first column (i.e all the 9 values of R, B and G are added)
					{
						// For 3 by 3 filter it will go to fisrt 3 Red pixels then the next 3 and then the next 3 , row wise, Imagine a square block
						if(count<filterW)
						{
							matcol2im.finalmatrix[0][j] = mat.finalmatrix[rowCount][colCount];
							j++;
							count++;
						}
						else
						{
							j = j + ((widthBefPad+2*pad)-filterW);
							count = 0;
						}
						rCount++;
	
					}
					else {
						
							j = rowInc + pixelsToMove;
							rCount = 0;
					}
				}
				
			}
		

		// Now get back the original image.

		NumJava matpad = originalmat; // dimension of an image before any padding was appliend on it.
		int counter=0;
		j=0;
		count=0;
		while (channelCount<channels)
		{
			for (int i=0;i<(imHeight+pad*2);i++)
			{
				// Not to put padded values.
				if(i<pad || i>=((imHeight+pad)))    // for first and last row padding 
				{
					while (count<(imWidth+pad*2))   // Do it for all columns 
					{
						//matpad.finalmatrix[0][j] = padval;
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
						//matpad.finalmatrix[0][j] = padval;
						count++;
						j++;
					}
					count=0;
					// put only non padded values.
					while(count<imWidth)    // Rest of the cols value of the old matrix
					{
						matpad.finalmatrix[0][counter] = matcol2im.finalmatrix[0][j];
						count++;
						j++;
						counter++;
						//orgcount++;
					}
					count=0;
					// Not to put padded values.
					while(count<pad)     // again for last cols padding.
					{

						//matpad.finalmatrix[0][j] = padval;
						count++;
						j++;
					}

					count=0;
				}
				//j++;
			}
			channelCount++; // For the next channel.

		}
		return matpad;
	}



	public static void main(String args[])
	{
		try 
		{


			NumJava x1=NumJava.creatematrix(new float[][]{{1,2,3},{4,5,6},{9,8,7}});
			NumJava x2=NumJava.creatematrix(new float[][]{{1,2,3},{4,5,6},{9,8,7}});
			NumJava.print(x1);

			System.out.println("Max element by Value :"+NumJava.get_Max_Element_Matrix_by_Value(x1));
			System.out.println("Max_Argument :"+NumJava.get_Arg_Max(x1)[0]+" "+NumJava.get_Arg_Max(x1)[1]);
			System.out.println("Max_element by row :"+NumJava.get_Max_Element_Matrix_by_Row(x1, 1));
			System.out.println("Max element by column :"+NumJava.get_Max_Element_Matrix_by_Column(x1,1));
			System.out.println("Min elemet by Value :"+NumJava.get_Min_Element_Matrix_by_Value(x1));
			System.out.println("Min elemet by Index :"+NumJava.get_Min_Element_Matrix_by_Index(x1)[0]+" "+NumJava.get_Min_Element_Matrix_by_Index(x1)[1]);
			System.out.println("Min elemet by row :"+NumJava.get_Min_Element_Matrix_by_Row(x1, 1));
			System.out.println("Min elemet by column :"+NumJava.get_Min_Element_Matrix_by_Column(x1, 1));
			NumJava.print(NumJava.transform_Matrix_with_Exponential(x1)); 
			NumJava.print(NumJava.make_Unit_Matrix(x1));   

		}

		catch(Exception e)
		{
			e.printStackTrace();
			//System.exit(0);
		}
	}

	// tested
	public void printMe() {
		System.out.println( finalmatrix.length + " " + finalmatrix[0].length);
		for (int i = 0; i < finalmatrix.length; i++) {
			for (int j = 0;j < finalmatrix[0].length; j++) {
				System.out.printf("%9.4f ", finalmatrix[i][j]);
			}
			System.out.println();    
		}    
	}	
}   
