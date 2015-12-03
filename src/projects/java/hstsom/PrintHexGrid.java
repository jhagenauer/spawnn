package hstsom;

import java.awt.Color;
import java.awt.Polygon;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Map;

import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;

import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;

public class PrintHexGrid {
	public static void main(String args[]) {
		Grid2DHex<double[]> grid = new Grid2DHex<double[]>(10, 6);

		GridPos c = new GridPos(4, 2);
		Map<GridPos, Color> p0 = new HashMap<GridPos, Color>();
		p0.put(c, Color.RED);
		p0.put(new GridPos(2, 1), Color.GREEN);

		for (GridPos p : grid.getPositions())
			if (grid.dist(p, c) <= 2 && !p0.containsKey(p))
				p0.put(p, Color.LIGHT_GRAY);

		saveHexEPS(grid, 5, p0, "output/geosom_idea.eps");

	}

	public static void saveHexEPS(Grid2DHex<double[]> grid, int scale, Map<GridPos, Color> p0, String fn) {
		try {
			int xDiff = 12;
			int yDiff = 14;

			int xDim = grid.getSizeOfDim(0);
			int yDim = grid.getSizeOfDim(1);

			FileOutputStream stream = new FileOutputStream(fn);
			EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
			g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
			g.setupDocument(stream, (int) Math.ceil((8 * xDim + 4 * (xDim + 1)) * scale), (int) Math.ceil(14 * yDim * scale + 7 * scale + 1));
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

			for (GridPos pos : grid.getPositions()) {
				int i = pos.getPos(0);
				int j = pos.getPos(1);
				int xc = i * xDiff + (int) (2 * xDiff * 1.0 / 3);
				int yc = j * yDiff + (int) (yDiff * 1.0 / 2);

				if (i % 2 == 1)
					yc += yDiff * 1.0 / 2;

				// (4|7), P3 (8|0), P5 (4|−7), P7 (−4|−7), P9 (−8|0), P11 (−4|7)
				int[] x = { xc + 4, xc + 8, xc + 4, xc - 4, xc - 8, xc - 4, xc + 4 };
				int[] y = { yc + 7, yc + 0, yc - 7, yc - 7, yc + 0, yc + 7, yc + 7 };

				Polygon p = new Polygon(x, y, x.length);

				AffineTransform at = new AffineTransform();
				at.scale(scale, scale);
				/*
				 * at.shear(-1.0, 0); at.translate(81, 0);
				 */
				Shape sp = at.createTransformedShape(p);

				if (p0.containsKey(pos)) {
					g.setColor(p0.get(pos));
					g.fill(sp);
				}

				g.setColor(Color.BLACK);
				g.draw(sp);
			}
			g.finish();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
