package context.space.binary_field;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.Drawer;

public class DoubleGrid2DUtils {
		
	public static void draw(Grid2D<Double> bf, String fn, boolean hiliCenter ) {
		int cellSize = 25*3;
		try {
			int minX = 0;
			int minY = 0;
			int maxX = 0;
			int maxY = 0;
			
			for( GridPos p : bf.getPositions() ) {
				int[] v = p.getPosVector();
				minX = Math.min(v[0], minX);
				maxX = Math.max(v[0], maxX);

				minY = Math.min(v[1], minY);
				maxY = Math.max(v[1], maxY);
			}

			BufferedImage bufImg = new BufferedImage(cellSize * (maxX - minX + 1)+1, cellSize * (maxY - minY + 1)+1, BufferedImage.TYPE_INT_ARGB);
			Graphics2D g2 = bufImg.createGraphics();
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			
			Map<GridPos,Double> values = new HashMap<GridPos,Double>();
			for( GridPos p : bf.getPositions() )
				values.put( p, bf.getPrototypeAt(p).doubleValue() );
			Map<GridPos,Color> colors = ColorBrewerUtil.valuesToColors(values, ColorMode.Blues);

			for( GridPos p : bf.getPositions() ) {
				int[] v = p.getPosVector();	
				
				g2.setColor( Drawer.getColor( bf.getPrototypeAt(p).floatValue() ) );
				//g2.setColor(colors.get(p));
			
				int x = (v[0] - minX) * cellSize;
				int y = (v[1] - minY) * cellSize;
				g2.fillRect( x, y, cellSize, cellSize);
				
				// outline
				g2.setColor(Color.GRAY);
				g2.drawRect( x, y, cellSize, cellSize);
				
				DecimalFormat df = new DecimalFormat("0.000");
				g2.setColor(Color.BLACK);
				g2.drawString(df.format(bf.getPrototypeAt(p))+"", x, y+10);
				//g2.drawString(p+"", x, y+10);
			}
			
			if( hiliCenter ) {
				g2.setColor(Color.GRAY);
				g2.fillOval( (int)((0 - minX + 0.5) * cellSize)-5, (int)((0 - minY + 0.5 ) * cellSize)-5, 10, 10);
			}
			ImageIO.write(bufImg, "png", new File(fn));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void saveAsEsriGrid(Grid2D<Double> bf, String fn) {
		int xSize = bf.getSizeOfDim(0);
		int ySize = bf.getSizeOfDim(1);
		
		int minX = 0;
		int minY = 0;
		int maxX = 0;
		int maxY = 0;
		
		for( GridPos p : bf.getPositions() ) {
			int[] v = p.getPosVector();
			minX = Math.min(v[0], minX);
			maxX = Math.max(v[0], maxX);

			minY = Math.min(v[1], minY);
			maxY = Math.max(v[1], maxY);
		}
				
		//DecimalFormat df = new DecimalFormat("0.000");
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(fn));

			bw.write("ncols\t" + xSize + "\n");
			bw.write("nrows\t" + ySize + "\n");
			bw.write("xllcorner\t0.0\n");
			bw.write("yllcorner\t0.0\n");
			bw.write("cellsize\t50.0\n");
			
			for (int i = minX; i <= maxX; i++) {
				for (int j = minY; j <= maxY; j++) {
					GridPos p = new GridPos(i,j);
					if( bf.getPositions().contains(p) ) {
						double v = bf.getPrototypeAt(p);
						bw.write( v + " ");
					} else {
						bw.write( -1.0 + " ");
					}
				}
				bw.write("\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try { bw.close(); } catch (Exception e) { }
		}
	}

	public static void saveAsCSV(Grid2D<Double> bf, String fn) {
		List<GridPos> s = new ArrayList<GridPos>(bf.getPositions());
		Collections.sort(s);

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(fn));
			bw.write("value,x,y\n");

			for (GridPos i : s) {
				int[] pos = i.getPosVector();
				double v = bf.getPrototypeAt(i);
				bw.write( v + "," + pos[0] + "," + pos[1] + "\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				bw.close();
			} catch (Exception e) {
			}
		}
	}
	
	public static void saveWeightMatrix(Grid2D<Double> bf, String fn) {
		List<GridPos> s = new ArrayList<GridPos>(bf.getPositions());
		Collections.sort(s);
		
		Map<GridPos,Integer> idxMap = new HashMap<GridPos,Integer>();
		for( int i = 0; i < s.size(); i++ )
			idxMap.put(s.get(i), i);

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(fn));
			bw.write("id1,id2,dist\n");

			for( GridPos p : s ) 
				for(GridPos nb : bf.getNeighbours(p) )
					bw.write( idxMap.get(p) + "," + idxMap.get(nb) + ",1\n");
			
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				bw.close();
			} catch (Exception e) {
			}
		}
	}
}
