package core;

public class convol {
	
	static int get_Row_Size(double [][] matrix){
		return matrix.length;
	}

	static int get_Column_Size(double [][] matrix) {
		return matrix[0].length;
	}

	static double get_Max_Element_Matrix_by_Value(double [][] matrix) throws Exception {

		if(matrix == null) {
			throw new Exception("Matrix is null");
		}

		double max = matrix[0][0];

		for (int row = 0; row < matrix.length; row++) {
			for (int column = 0; column < matrix[0].length; column++) {
				if(matrix[row][column] > max) {
					max = matrix[row][column];
				}
			}
		}
		return max;
	}

	// Returns an array 1st index is row and 2nd is column
	static int[] get_Arg_Max(double [][] matrix) throws Exception {

		if(matrix == null) {
			throw new Exception("Matrix is null");
		}

		double max = matrix[0][0];
		
		int[] index_Max = new int[2];

		for (int row = 0; row < matrix.length; row++) {
			for (int column = 0; column < matrix[0].length; column++) {
				if(matrix[row][column] > max) {
					max = matrix[row][column];
					index_Max[0] = row;
					index_Max[1] = column;
				}
			}
		}
		return index_Max;
	}

	static double get_Max_Element_Matrix_by_Row(double [][] matrix, int row_Number) {

		double max = matrix[row_Number][0];

		for (int row = 0; row < matrix.length; row++) {
			if(matrix[row_Number][row] > max) {
				max = matrix[row_Number][row];
			}
		}
		return max;
	}

	static double get_Max_Element_Matrix_by_Column(double [][] matrix, int column_Number) {

		double max = matrix[0][column_Number];

		for (int column = 0; column < matrix[0].length; column++){
			if(matrix[column][column_Number] > max) {
				max = matrix[column][column_Number];
			}
		}
		return max;
	}
	
	static double get_Min_Element_Matrix_by_Value(double [][] matrix) throws Exception {

		if(matrix == null) {
			throw new Exception("Matrix is null");
		}

		double min = matrix[0][0];

		for (int row = 0; row < matrix.length; row++) {
			for (int column = 0; column < matrix[0].length; column++) {
				if(matrix[row][column] < min) {
					min = matrix[row][column];
				}
			}
		}
		return min;
	}

	// Returns an array 1st index is row and 2nd is column
	static int[] get_Min_Element_Matrix_by_Index(double [][] matrix) throws Exception {

		if(matrix == null) {
			throw new Exception("Matrix is null");
		}

		double min = matrix[0][0];
		
		int[] index_Min = new int[2];

		for (int row = 0; row < matrix.length; row++) {
			for (int column = 0; column < matrix[0].length; column++) {
				if(matrix[row][column] < min) {
					min = matrix[row][column];
					index_Min[0] = row;
					index_Min[1] = column;
				}
			}
		}
		return index_Min;
	}

	static double get_Min_Element_Matrix_by_Row(double [][] matrix, int row_Number) {

		double min = matrix[row_Number][0];

		for (int row = 0; row < matrix.length; row++) {
			if(matrix[row_Number][row] > min) {
				min = matrix[row_Number][row];
			}
		}
		return min;
	}

	static double get_Min_Element_Matrix_by_Column(double [][] matrix, int column_Number) {

		double min = matrix[0][column_Number];

		for (int column = 0; column < matrix[0].length; column++){
			if(matrix[column][column_Number] > min) {
				min = matrix[column][column_Number];
			}
		}
		return min;
	}

	static double calculate_Exponential(double elem) {
		return Math.exp(elem);
	}
	
	// transformed with the matrix reference
	static double [][] transform_Matrix_with_Exponential(double [][] matrix) {
		
		for (int row = 0; row < matrix.length; row++) {
			for (int column = 0; column < matrix[0].length; column++) {
				matrix[row][column] = calculate_Exponential(matrix[row][column]);
			}
		}
		return matrix;
	}
	
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
	
	static double [][] make_Unit_Matrix(double [][] matrix) {
		
		for (int row = 0; row < matrix.length; row++) {
			for (int column = 0; column < matrix[0].length; column++) {
					matrix[row][column] = 1.0;
				}
			}
		
		return matrix;
	}
}
