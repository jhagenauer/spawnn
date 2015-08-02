package hstsom;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;

public class PrintHexGrid {
	public static void main(String args[]) {
		Grid2DHex<double[]> grid = new Grid2DHex<double[]>( 10, 6 );
		
		try {
			GridPos p0 = new GridPos(1,1);
			GridPos p1 = new GridPos(8,4);
			GridPos p2 = new GridPos(4,2);
			
			SomUtils.printImage(getHexClassDistImage( p0, grid, 5), new FileOutputStream("output/grid_0.png"));
			SomUtils.printImage(getHexClassDistImage( p1, grid, 5), new FileOutputStream("output/grid_1.png"));
			SomUtils.printImage(getHexClassDistImage( p2, grid, 5), new FileOutputStream("output/grid_2.png"));
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}		
	}
	
	public static BufferedImage getHexClassDistImage( GridPos p0, Grid2DHex<double[]> grid, int scale ) {
		int xDiff = 12;
		int yDiff = 14;

		BufferedImage bufImg = new BufferedImage( 980, 267 , BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				
		for (GridPos pos : grid.getPositions() ) {				
			int i = pos.getPosVector()[0];
			int j = pos.getPosVector()[1];
			int xc = i * xDiff + (int)(2*xDiff*1.0/3);
			int yc = j * yDiff + (int)(yDiff*1.0/2);
			
			if( i % 2 == 1 ) 
				yc += yDiff*1.0/2;
			
			// (4|7), P3 (8|0), P5 (4|−7), P7 (−4|−7), P9 (−8|0), P11 (−4|7)
			int[] x = { xc+4, xc+8, xc+4, xc-4, xc-8, xc-4, xc+4 };
			int[] y = { yc+7, yc+0, yc-7, yc-7, yc+0, yc+7, yc+7 };
			
			Polygon p = new Polygon(x, y, x.length);	
		  			
			AffineTransform at = new AffineTransform();
		    at.scale(scale,scale/1.75);
		    at.shear(-1.0, 0);
		    at.translate(81, 0);
		    Shape sp = at.createTransformedShape(p);
		    
		    if( pos.equals(p0) ) {
		    	g2.setColor(Color.GRAY);
		    	g2.fill(sp);
		    }
		    		    
		    g2.setColor( Color.BLACK );
		    g2.draw(sp);
		}
		g2.dispose();
		return bufImg;
	}
}
