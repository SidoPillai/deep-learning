package test;

import java.util.Random;

import core.Convol;


public class ConvolTest {
	
	static Random random = new Random();

	public static void main(String[] args) {
	
		double [][] matrix = constructMatrixWithRandomElements(4,4);
		
		printMatrix(matrix);
		
		Convol c = new Convol();
		
		System.out.println("Testing the Get Row Size");
		System.out.println(c.get_Row_Size(matrix));
		System.out.println("Testing the Get Column Size");
		System.out.println(c.get_Column_Size(matrix));
		
		
		
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
