package spawnn.gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.Stroke;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;

public class GridPanel extends NeuronVisPanel<GridPos> implements MouseListener {
	
	private static Logger log = Logger.getLogger(GridPanel.class);
	private static final long serialVersionUID = 5760512747937681233L;
					
	int maxX = Integer.MIN_VALUE, minX = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE, minY = Integer.MAX_VALUE;
	Map<GridPos,Polygon> cells = new HashMap<GridPos,Polygon>();
	
	Map<GridPos,Shape> scaledCells = new HashMap<GridPos,Shape>();
	Map<GridPos,Color> gridColors = new HashMap<GridPos,Color>();
	Map<GridPos,Color> selectedColors = new HashMap<GridPos,Color>();
	
	public GridPanel(Grid2D<double[]> grid, Dist<double[]> fDist) {		
		for( GridPos gp : grid.getPositions() ) {
			int i = gp.getPos(0);
			int j = gp.getPos(1);
				
				int[] x = null, y = null;
				
				if( grid instanceof Grid2DHex ) {
					int xDiff = 12;
					int yDiff = 14;
					
					int xc = i * xDiff + (int) (2 * xDiff * 1.0 / 3);
					int yc = j * yDiff + (int) (yDiff * 1.0 / 2);

					if (i % 2 == 1)
						yc += yDiff * 1.0 / 2;

					x = new int[]{ xc + 4, xc + 8, xc + 4, xc - 4, xc - 8, xc - 4, xc + 4 };
					y = new int[]{ yc + 7, yc + 0, yc - 7, yc - 7, yc + 0, yc + 7, yc + 7 };
				
				} else {
					int xc = i * 12;
					int yc = j * 12;
										
					x = new int[]{ xc, xc + 12, xc + 12, xc +  0, xc };
					y = new int[]{ yc, yc +  0, yc + 12, yc + 12, yc };
					
				}
				
				for (int k : x ) {
					minX = Math.min(k, minX);
					maxX = Math.max(k, maxX);
				}

				for (int k : y ) {
					minY = Math.min(k, minY);
					maxY = Math.max(k, maxY);
				}

				Polygon p = new Polygon(x, y, x.length);
				cells.put( gp, p);	
		}
	}
	
	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		updateScaledCells();
		draw( (Graphics2D)g);
	}
	
	private void draw( Graphics2D g2 ) {
		Stroke origStroke = g2.getStroke();
		BasicStroke fatStroke = new BasicStroke(SELECTED_WIDTH);
		
		// "clear" old border
		g2.setStroke(fatStroke);		
		for( GridPos p : scaledCells.keySet() ) {
			g2.setColor( getBackground() );
			g2.draw( scaledCells.get(p));
		}
		g2.setStroke(origStroke);
			
		// fill
		for( GridPos p : scaledCells.keySet() ) {
			if( gridColors.containsKey(p) ) 
				g2.setColor( gridColors.get( p ) );
			else
				g2.setColor( Color.GRAY );
			g2.fill( scaledCells.get(p) );
		}
		
		// draw border
		for( GridPos p : scaledCells.keySet() ) {
			g2.setColor( Color.BLACK );
			g2.draw( scaledCells.get(p));
			
			/*int x = (int)scaledCells.get(p).getBounds().getCenterX();
			int y = (int)scaledCells.get(p).getBounds().getCenterY();
			g2.drawString(p.toString(), x, y);*/
		}
		
		// fill selected
		for( GridPos p : selectedColors.keySet() ) {
			Color c = selectedColors.get(p);
			g2.setColor(new Color( c.getRed(), c.getGreen(), c.getBlue(), NeuronVisPanel.SELECTED_OPACITY ));
			g2.setStroke(fatStroke);
			g2.fill( scaledCells.get(p) );
		}
				
		// draw border selected
		g2.setStroke(fatStroke);
		for( GridPos p : selectedColors.keySet() ) {
			g2.setColor( selectedColors.get(p));
			g2.setStroke(fatStroke);
			g2.draw( scaledCells.get(p) );
		}
		g2.setStroke(origStroke);
	}
	
	@Deprecated
	public BufferedImage getBufferedImage() {
		BufferedImage bufImg = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_ARGB);
		
		Graphics2D g = bufImg.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		draw(g);
		
		g.dispose();
		return bufImg;
	}
	
	@Override
	public void saveImage( File fn, String mode ) {
		try {		
			FileOutputStream stream = new FileOutputStream(fn);			
			if( mode.equals("PNG") ) {
				BufferedImage bufImage = new BufferedImage((int)Math.ceil(maxX*scale)+2*offset, (int)Math.ceil(maxY*scale)+2*offset, BufferedImage.TYPE_INT_ARGB);
				Graphics2D g = bufImage.createGraphics();

				g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				draw(g);
				
				ImageIO.write(bufImage, "PNG", stream);
			} else if( mode.equals("EPS")) {
				EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
				g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
				g.setupDocument(stream, (int)Math.ceil(scale*maxX+2*offset), (int)Math.ceil(scale*maxY+2*offset) ); 	
								
				draw(g);
				g.finish();
			} else {
				log.debug("Unknown file format!");
			}
			stream.flush();
			stream.close();
		} catch( Exception e ) {
			e.printStackTrace();
		}
	}
	

	private double scale = 1.0;
	private final int offset = (int)Math.ceil((1.0+SELECTED_WIDTH)/2);
	
	public void updateScaledCells() {
		scaledCells = new HashMap<GridPos,Shape>();
		for (GridPos p : cells.keySet() ) {
			AffineTransform at = new AffineTransform();
			scale = Math.min((double) ( getWidth() - 2*offset )/ (maxX - minX), (double) ( getHeight()- 2*offset )/ (maxY - minY));
			at.translate(offset, offset);
			at.scale( scale, scale );
			Shape sp = at.createTransformedShape(cells.get(p));
			scaledCells.put(p, sp);
		}
	}
	
	@Override
	public void setColors(Map<GridPos, Color> gridColors, Map<GridPos, Color> selectedColors, Map<GridPos,Double> neuronValues ) {
		this.gridColors = gridColors;
		this.selectedColors = selectedColors;
		repaint();
	}
	
	@Override
	public void mouseClicked(MouseEvent e) {
		for (GridPos p : cells.keySet() ) 
			if( scaledCells.get(p).contains(e.getX(), e.getY())) {
				fireNeuronSelectedEvent( new NeuronSelectedEvent<GridPos>(this, p));
				break;
			}
	}

	@Override
	public void mouseEntered(MouseEvent e) {
	}

	@Override
	public void mouseExited(MouseEvent e) {
	}

	@Override
	public void mousePressed(MouseEvent e) {
	}

	@Override
	public void mouseReleased(MouseEvent e) {	
	}
}
