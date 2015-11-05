package test;

import java.util.Random;

import core.Convol;


public class ConvolTest {

	static Random random = new Random();

	public static void main(String[] args) throws Exception {

		double [][] matrix = constructMatrixWithRandomElements(4,4);

		printMatrix(matrix);

		Convol c = new Convol();

		System.out.println("Testing the Get Row Size");
		System.out.println(c.get_Row_Size(matrix));
		System.out.println("Testing the Get Column Size");
		System.out.println(c.get_Column_Size(matrix));
		System.out.println("Get max element");
		System.out.println(c.get_Max_Element_Matrix_by_Value(matrix));
		System.out.println("Get Arg Max");
		int[] max = c.get_Arg_Max(matrix);
		System.out.println("i " + max[0] + " j " + max[1]);
		System.out.println("Get max element by row");
		System.out.println(" 1 " + c.get_Max_Element_Matrix_by_Row(matrix, 1));
		System.out.println(" 2 " + c.get_Max_Element_Matrix_by_Row(matrix, 2));
		System.out.println("Get max element by column");
		System.out.println(" 1 " + c.get_Max_Element_Matrix_by_Column(matrix, 1));
		System.out.println(" 2 " + c.get_Max_Element_Matrix_by_Column(matrix, 2));
		System.out.println("Get min element");
		System.out.println(c.get_Min_Element_Matrix_by_Value(matrix));
		System.out.println("Get min element by row");
		System.out.println(" 1 " + c.get_Min_Element_Matrix_by_Row(matrix, 1));
		System.out.println(" 2 " + c.get_Min_Element_Matrix_by_Row(matrix, 2));
		System.out.println("Get min element by column");
		System.out.println(" 1 " + c.get_Min_Element_Matrix_by_Column(matrix, 1));
		System.out.println(" 2 " + c.get_Min_Element_Matrix_by_Column(matrix, 2));
		System.out.println(" Get Arg Min");
		int[] min = c.get_Min_Element_Matrix_by_Index(matrix);
		System.out.println("i " + min[0] + " j " + min[1]);
		System.out.println("Calculate exponential of matrix[2][3]" + c.calculate_Exponential(matrix[2][3]));
		double[][] exponentialMatrix = c.transform_Matrix_with_Exponential(matrix);
		printMatrix(exponentialMatrix);


	}

	public static double[][] constructMatrixWithRandomElements(int row, int column) {
		double [][] matrix = new double[row][column];

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				matrix[i][j] = random.nextDouble(); 
			}
		}
		return matrix;
	}

	public static double[][] printMatrix(double [][] matrix) {

		int row = matrix.length;
		int column = matrix[0].length;

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
		return matrix;
	}

}
