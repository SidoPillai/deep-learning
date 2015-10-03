import java.util.*;
class numjava
{
    private int M;
    private int N;
    private float finalmatrix [][];
    public numjava(int M,int N)
    {
        this.M=M;
        this.N=N;
        finalmatrix=new float[M][N];
    }
    
    // To create random 2-d array
    public static numjava createrandom (int M,int N)
    {
        numjava mat=new numjava(M,N);
        for (int i=0;i<M;i++)
        {
            for (int j=0;j<N;j++)
            {
                mat.finalmatrix[i][j]=(float)Math.random();
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
                System.out.printf("%9.4f ", mat.finalmatrix[i][j]);
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
            for (int j=0;j<N1;j++)
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
    
    //Elementwise multiplication
    public static numjava elementmul(numjava mat1,numjava mat2)
    {
        int M1=mat1.M;
        int N1=mat1.N;
        int M2=mat2.M;
        int N2=mat2.N;
        try
        {
            if (N1!=M2)
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
    public static void shape(float mat1[][])
    {
       return mat1[0].length,mat1[1].length;     
    }
    */
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
              
              
          //}    
         /*
         catch(Exception e)
         {
         }
         */       
         return C;       
         //   System.out.println("Cannot rehape matrix");
           
    }
    
    // Creating a matrix
    public static numjava creatematrix(float mat[][])
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
//    int counteri=0;
//    int counterj=0;
    for (int i=0;i<mat.M;i++)
    {
        for (int j=0;j<mat.N;j++)
        {
                   
            C.finalmatrix[j][i]= mat.finalmatrix[i][j];
            
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
	static float get_Max_Element_Matrix_by_Value(numjava matrix) throws Exception {

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

	// Returns an array 1st index is row and 2nd is column
	static int[] get_Arg_Max(numjava matrix) throws Exception {

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

	static float get_Max_Element_Matrix_by_Row(numjava matrix, int row_Number) {

		float max = matrix.finalmatrix[row_Number][0];

		for (int row = 0; row < matrix.M; row++) {
			if(matrix.finalmatrix[row_Number][row] > max) {
				max = matrix.finalmatrix[row_Number][row];
			}
		}
		return max;
	}

	static float get_Max_Element_Matrix_by_Column(numjava matrix, int column_Number) {

		float max = matrix.finalmatrix[0][column_Number];

		for (int column = 0; column < matrix.N; column++){
			if(matrix.finalmatrix[column][column_Number] > max) {
				max = matrix.finalmatrix[column][column_Number];
			}
		}
		return max;
	}
	
	static float get_Min_Element_Matrix_by_Value(numjava matrix) throws Exception {

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

	// Returns an array 1st index is row and 2nd is column
	static int[] get_Min_Element_Matrix_by_Index(numjava matrix) throws Exception {

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

	static float get_Min_Element_Matrix_by_Row(numjava matrix, int row_Number) {

	 float min = matrix.finalmatrix[row_Number][0];

		for (int row = 0; row < matrix.M; row++) {
			if(matrix.finalmatrix[row_Number][row] > min) {
				min = matrix.finalmatrix[row_Number][row];
			}
		}
		return min;
	}

	static float get_Min_Element_Matrix_by_Column(numjava matrix, int column_Number) {

		float min = matrix.finalmatrix[0][column_Number];

		for (int column = 0; column < matrix.N; column++){
			if(matrix.finalmatrix[column][column_Number] > min) {
				min = matrix.finalmatrix[column][column_Number];
			}
		}
		return min;
	}

	static float calculate_Exponential(float elem) {
		return (float)Math.exp(elem);
	}
	
	// transformed with the matrix reference
	static  numjava transform_Matrix_with_Exponential(numjava matrix) {
		
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
	
	static numjava make_Unit_Matrix(numjava matrix) {
		
		for (int row = 0; row < matrix.M; row++) {
			for (int column = 0; column < matrix.N; column++) {
					matrix.finalmatrix[row][column] = 1;
				}
			}
		
		return matrix;
	}
          
        /**
     * Adding two matrices
     * @param mat1 
     * @param mat2
     * @return
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
		    		for (int i=0;i<mat1.M;i++)
		    		{
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
     /*
	* To get sum of elements along row or column
	* axis = 0 : sum along row 
	* axis = 1 : sum along column
	*/
	public static numjava sum(numjava mat1, int axis)
	{
		numjava mat = null;
		float sum = 0;
		try 
		{

			if (mat1.finalmatrix!=null)
			{
				if (axis == 0)
				{

					// sum along row axis
					mat = new numjava(1,mat1.N);
					for (int j = 0 ;j<mat1.M ; j++)
					{
						for (int i = 0; i<mat1.N; i++)
						{
							sum = sum + mat1.finalmatrix[i][j];	
						}
						mat.finalmatrix[0][j] = sum;
					}

				}
				else if (axis == 1) 
				{
					// add along column axis

					mat = new numjava(mat1.M,1);
					for (int i = 0; i< mat1.M; i++)
					{
						for (int j = 0; j<mat1.N; j++)
						{
							sum = sum + mat1.finalmatrix[i][j]
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

  
          
          
    public static void main(String args[])
    {
        try 
        {
        
        
        numjava x1=numjava.creatematrix(new float[][]{{1,2,3},{4,5,6},{9,8,7}});
        numjava x2=numjava.creatematrix(new float[][]{{1,2,3},{4,5,6},{9,8,7}});
        numjava.print(x1);
       
        System.out.println("Max element by Value :"+numjava.get_Max_Element_Matrix_by_Value(x1));
        System.out.println("Max_Argument :"+numjava.get_Arg_Max(x1)[0]+" "+numjava.get_Arg_Max(x1)[1]);
        System.out.println("Max_element by row :"+numjava.get_Max_Element_Matrix_by_Row(x1, 1));
        System.out.println("Max element by column :"+numjava.get_Max_Element_Matrix_by_Column(x1,1));
        System.out.println("Min elemet by Value :"+numjava.get_Min_Element_Matrix_by_Value(x1));
        System.out.println("Min elemet by Index :"+numjava.get_Min_Element_Matrix_by_Index(x1)[0]+" "+numjava.get_Min_Element_Matrix_by_Index(x1)[1]);
        System.out.println("Min elemet by row :"+numjava.get_Min_Element_Matrix_by_Row(x1, 1));
        System.out.println("Min elemet by column :"+numjava.get_Min_Element_Matrix_by_Column(x1, 1));
        numjava.print(numjava.transform_Matrix_with_Exponential(x1)); 
        numjava.print(numjava.make_Unit_Matrix(x1));   
     
        }
        
        catch(Exception e)
        {
        e.printStackTrace();
        //System.exit(0);
        }
    }	
}   
