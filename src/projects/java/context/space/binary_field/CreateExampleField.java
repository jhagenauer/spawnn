package context.space.binary_field;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;

public class CreateExampleField {

	public static void main(String[] args) {

		{
			Grid2D<Double> df = new Grid2D<Double>(0,0);
			df.setPrototypeAt(new GridPos(0, 0), 1.0);
			df.setPrototypeAt(new GridPos(1, 0), 0.35);
			df.setPrototypeAt(new GridPos(0, 1), 0.75);
			df.setPrototypeAt(new GridPos(-1, 0), 0.5);
			df.setPrototypeAt(new GridPos(0, -1), 0.65);

			df.setPrototypeAt(new GridPos(-1, -1), -1.0);
			df.setPrototypeAt(new GridPos(1, 1), -1.0);
			df.setPrototypeAt(new GridPos(-1, 1), -1.0);
			df.setPrototypeAt(new GridPos(1, -1), -1.0);
		}
		
		{ // 1
			Grid2D<Double> df = new Grid2D<Double>(0,0);
			for( int i = -2; i <= 2; i++)
				for( int j = -2; j <= 2; j++ )
					df.setPrototypeAt(new GridPos(i, j), -1.0);
			
			df.setPrototypeAt(new GridPos(0, 0), 1.0);
			
		}
		
		{ // 2
			Grid2D<Double> df = new Grid2D<Double>(0,0);
			for( int i = -2; i <= 2; i++)
				for( int j = -2; j <= 2; j++ )
					df.setPrototypeAt(new GridPos(i, j), -1.0);

			df.setPrototypeAt(new GridPos(1, 0), 0.3);
			df.setPrototypeAt(new GridPos(0, 1), 0.75);
			df.setPrototypeAt(new GridPos(-1, 0), 0.8);
			df.setPrototypeAt(new GridPos(0, -1), 0.65);
			
		}
		
		{ // 3
			Grid2D<Double> df = new Grid2D<Double>(0,0);
			for( int i = -2; i <= 2; i++)
				for( int j = -2; j <= 2; j++ )
					df.setPrototypeAt(new GridPos(i, j), -1.0);

			df.setPrototypeAt(new GridPos(2, 0), 0.45);
			df.setPrototypeAt(new GridPos(0, 2), 0.55);
			df.setPrototypeAt(new GridPos(-2, 0), 0.5);
			df.setPrototypeAt(new GridPos(0, -2), 0.55);
			
			df.setPrototypeAt(new GridPos(-1, 1), 0.55);
			df.setPrototypeAt(new GridPos(1, 1), 0.45);
			df.setPrototypeAt(new GridPos(1, -1), 0.5);
			df.setPrototypeAt(new GridPos(-1, -1), 0.5);
			
		}
	}

}
