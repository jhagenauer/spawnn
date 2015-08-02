package context.space.binary_field;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DToroid;
import spawnn.som.grid.GridPos;

public class BooleanGrid2DUtils {
	
	private static Logger log = Logger.getLogger(BooleanGrid2DUtils.class);

	public static void draw(Grid2D<Boolean> bf, String fn, boolean hiliCenter) {
		int cellSize = 25; //25
		try {
			int minX = 0;
			int minY = 0;
			int maxX = 0;
			int maxY = 0;

			for (GridPos p : bf.getPositions()) {
				int[] v = p.getPosVector();
				minX = Math.min(v[0], minX);
				maxX = Math.max(v[0], maxX);

				minY = Math.min(v[1], minY);
				maxY = Math.max(v[1], maxY);
			}

			BufferedImage bufImg = new BufferedImage(cellSize * (maxX - minX + 1) + 1, cellSize * (maxY - minY + 1) + 1, BufferedImage.TYPE_INT_ARGB);
			Graphics2D g2 = bufImg.createGraphics();
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

			for (GridPos p : bf.getPositions()) {
				int[] v = p.getPosVector();
				if (bf.getPrototypeAt(p))
					g2.setColor(Color.BLACK);
				else
					g2.setColor(Color.WHITE);

				int x = (v[0] - minX) * cellSize;
				int y = (v[1] - minY) * cellSize;
				g2.fillRect(x, y, cellSize, cellSize);

				// outline
				g2.setColor(Color.GRAY);
				g2.drawRect(x, y, cellSize, cellSize);

				//g2.setColor(Color.RED);
				//g2.drawString(p+"", x, y+10);
			}

			if (hiliCenter) {
				g2.setColor(Color.GRAY);
				g2.fillOval((int) ((0 - minX + 0.5) * cellSize) - 5, (int) ((0 - minY + 0.5) * cellSize) - 5, 10, 10);
			}
			ImageIO.write(bufImg, "png", new File(fn));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void drawEPS(Grid2D<Boolean> bf, String fn, boolean hiliCenter) {
		int cellSize = 25; //25
		try {
			int minX = 0;
			int minY = 0;
			int maxX = 0;
			int maxY = 0;

			for (GridPos p : bf.getPositions()) {
				int[] v = p.getPosVector();
				minX = Math.min(v[0], minX);
				maxX = Math.max(v[0], maxX);

				minY = Math.min(v[1], minY);
				maxY = Math.max(v[1], maxY);
			}

			//BufferedImage bufImg = new BufferedImage(cellSize * (maxX - minX + 1) + 1, cellSize * (maxY - minY + 1) + 1, BufferedImage.TYPE_INT_ARGB);
			//Graphics2D g2 = bufImg.createGraphics();
			//g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			
			FileOutputStream stream = new FileOutputStream(fn);
			EPSDocumentGraphics2D g2 = new EPSDocumentGraphics2D(false);
			g2.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext() );
			g2.setupDocument(stream, cellSize * (maxX - minX + 1) + 1, cellSize * (maxY - minY + 1) + 1);

			for (GridPos p : bf.getPositions()) {
				int[] v = p.getPosVector();
				if (bf.getPrototypeAt(p))
					g2.setColor(Color.BLACK);
				else
					g2.setColor(Color.WHITE);

				int x = (v[0] - minX) * cellSize;
				int y = (v[1] - minY) * cellSize;
				g2.fillRect(x, y, cellSize, cellSize);

				// outline
				g2.setColor(Color.GRAY);
				g2.drawRect(x, y, cellSize, cellSize);

				//g2.setColor(Color.RED);
				//g2.drawString(p+"", x, y+10);
			}

			if (hiliCenter) {
				g2.setColor(Color.GRAY);
				g2.fillOval((int) ((0 - minX + 0.5) * cellSize) - 5, (int) ((0 - minY + 0.5) * cellSize) - 5, 10, 10);
			}
			
			g2.finish();
			stream.flush();
			stream.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void saveAsEsriGrid(Grid2D<Boolean> bf, String fn) {
		int xSize = bf.getSizeOfDim(0);
		int ySize = bf.getSizeOfDim(1);

		int minX = 0;
		int minY = 0;
		int maxX = 0;
		int maxY = 0;

		for (GridPos p : bf.getPositions()) {
			int[] v = p.getPosVector();
			minX = Math.min(v[0], minX);
			maxX = Math.max(v[0], maxX);

			minY = Math.min(v[1], minY);
			maxY = Math.max(v[1], maxY);
		}

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

					GridPos p = new GridPos(i, j);
					int v;
					if (bf.getPositions().contains(p))
						v = bf.getPrototypeAt(p) ? 1 : 0;
					else
						v = -1;
					bw.write(v + " ");
				}
				bw.write("\n");
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

	public static void saveAsCSV(Grid2D<Boolean> bf, String fn) {
		List<GridPos> s = new ArrayList<GridPos>(bf.getPositions());
		Collections.sort(s);

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(fn));
			bw.write("value,x,y\n");

			for (GridPos i : s) {
				int[] pos = i.getPosVector();
				int v = bf.getPrototypeAt(i) ? 1 : 0;
				bw.write(v + "," + pos[0] + "," + pos[1] + "\n");
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

	public static void saveWeightMatrix(Grid2D<Boolean> bf, String fn) {
		List<GridPos> s = new ArrayList<GridPos>(bf.getPositions());
		Collections.sort(s);

		Map<GridPos, Integer> idxMap = new HashMap<GridPos, Integer>();
		for (int i = 0; i < s.size(); i++)
			idxMap.put(s.get(i), i);

		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(fn));
			bw.write("id1,id2,dist\n");

			for (GridPos p : s)
				for (GridPos nb : bf.getNeighbours(p))
					bw.write(idxMap.get(p) + "," + idxMap.get(nb) + ",1\n");

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				bw.close();
			} catch (Exception e) {
			}
		}
	}

	public static void buildBinaryGridTestData() {
		Random r = new Random();

		int xDim = 100, yDim = 100;
		Grid2D<Boolean> bf = new Grid2D<Boolean>(xDim, yDim);

		for (int i = 0; i < xDim; i++)
			for (int j = 0; j < yDim; j++)
				bf.setPrototypeAt(new GridPos(i, j), false);

		for (GridPos p : bf.getPositions()) {
			int nbT = 0;
			for (GridPos nb : bf.getNeighbours(p))
				if (bf.getPrototypeAt(nb))
					nbT++;

			boolean state = false;
			if ((nbT > 0 && r.nextDouble() < 0.7) || (nbT == 0 && r.nextDouble() < 0.3))
				state = true;

			bf.setPrototypeAt(p, state);
		}

		int trues = 0;
		for (boolean b : bf.getPrototypes())
			if (b)
				trues++;
		System.out.println("trues: " + trues + ", falses: " + (bf.size() - trues));

		BooleanGrid2DUtils.saveAsEsriGrid(bf, "output/grid" + xDim + "x" + yDim + ".asc");
		BooleanGrid2DUtils.saveAsCSV(bf, "output/grid" + xDim + "x" + yDim + ".csv");
		BooleanGrid2DUtils.saveWeightMatrix(bf, "output/grid" + xDim + "x" + yDim + ".wtg");
		BooleanGrid2DUtils.draw(bf, "output/grid" + xDim + "x" + yDim + ".png", false);
	}
	
	public static void buildBinaryGridTest(int xDim, int yDim, int mode, boolean toroid ) {

		Random r = new Random();
		String st;
		Grid2D<Boolean> bf;
		if( toroid ) {
			bf = new Grid2DToroid<Boolean>(xDim, yDim);
			st = "toroid";
		} else {
			bf = new Grid2D<Boolean>(xDim, yDim);
			st = "grid";
		}
		List<GridPos> falses = new ArrayList<GridPos>();
		Map<GridPos,Integer> nbs = new HashMap<GridPos,Integer>(); 
		
		for (int i = 0; i < xDim; i++) {
			for (int j = 0; j < yDim; j++) {
				GridPos p = new GridPos(i,j);
				bf.setPrototypeAt(p, false);
				falses.add(p);	
				nbs.put(p,1);
			}
		}
		
		// tournament selection
		while( falses.size() != xDim*yDim/2 ) {
						
			int sum = 0;
			for( GridPos p : falses )
				sum += nbs.get(p);
				
			int rnd = r.nextInt(sum) + 1;
			int from = 1, to;
			for( GridPos p : falses ) {
				to = from + nbs.get(p);
				
				if( from <= rnd && rnd < to ) { 
					bf.setPrototypeAt(p, true);
					falses.remove(p);
					
					Set<GridPos> toUpdate = new HashSet<GridPos>();
					toUpdate.add(p);
					toUpdate.addAll(bf.getNeighbours(p));
					
					for( GridPos tp : toUpdate) {
						int i = 0;
						for( GridPos nb : bf.getNeighbours(tp ) )
							if( bf.getPrototypeAt(nb) ) // count positive neighbors
								i++;
						if( mode == 0) {
							// nbs.put(tp, i+1); // 1 <= i <= 5
							nbs.put( tp, 1);
						} else if( mode == 1 )
							nbs.put(tp, (int)Math.pow(2, i ) ); // 1,2,4,8,16
						else if( mode == 2 )
							nbs.put(tp, (int)Math.pow(4, i ) ); // 1,2,4,8,16
					}
					
					break;
				}
				from = to;
			}
		}

		int trues = 0;
		for (boolean b : bf.getPrototypes())
			if (b)
				trues++;
		log.debug("trues: " + trues + ", falses: " + (bf.size() - trues));

		
		BooleanGrid2DUtils.saveAsEsriGrid(bf, "output/" + st + xDim + "x" + yDim + "_"+mode+".asc");
		BooleanGrid2DUtils.saveAsCSV(bf, "output/" + st + xDim + "x" + yDim + "_"+mode+".csv");
		BooleanGrid2DUtils.saveWeightMatrix(bf, "output/" + st + xDim + "x" + yDim + "_" +mode+ ".wtg");
		BooleanGrid2DUtils.draw(bf, "output/" + st + xDim + "x" + yDim + "_" +mode+ ".png", false);
		BooleanGrid2DUtils.drawEPS(bf, "output/" + st + xDim + "x" + yDim + "_" +mode+ ".eps", false);
	}

	public static void main(String[] args) {
		//buildBinaryGridTest(50,50,1,false);
		buildBinaryGridTest(50,50,0,true);
		buildBinaryGridTest(50,50,1,true);
	}
}
