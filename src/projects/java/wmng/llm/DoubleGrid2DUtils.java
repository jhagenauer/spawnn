package wmng.llm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DToroid;
import spawnn.som.grid.GridPos;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.Drawer;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

public class DoubleGrid2DUtils {
	
	private static Logger log = Logger.getLogger(DoubleGrid2DUtils.class);
	
	public static Grid2D<double[]> create1stOrderDoubleGrid(int xDim, int yDim, int numClust, boolean toroid ) {

		Random r = new Random();
		Grid2D<double[]> grid;
		if( toroid )
			grid = new Grid2DToroid<double[]>(xDim, yDim);
		else
			grid = new Grid2D<double[]>(xDim, yDim);
		
		List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions() );
		Collections.shuffle(pos);
		
		Set<GridPos> clustCenter = new HashSet<GridPos>();
		while( clustCenter.size() < numClust ) 
			clustCenter.add( pos.get(r.nextInt(pos.size() ) ) );
		
		Map<GridPos,Double> probSelect = new HashMap<GridPos,Double>();
		for( GridPos p : pos ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			grid.setPrototypeAt(p, new double[]{i,j,0,0});
						
			double d = Double.MAX_VALUE;
			for( GridPos c : clustCenter )
				d = Math.min(d, grid.dist(p,c));
			
			probSelect.put(p, d);
		}
		
		// inv and normalize probabilities
		double max = Collections.max(probSelect.values());
		for( GridPos p : pos )
			probSelect.put(p, Math.pow( (max - probSelect.get(p) )/max, 3) );
						
		double sumProb = 0;
		for( double d : probSelect.values() )
			sumProb += d;
								
		while( true ) {
			double rnd = r.nextDouble()*sumProb;
			double from = 0, to;
			
			for( GridPos p : pos ) {
				to = from + probSelect.get(p);
				
				if( from <= rnd && rnd < to ) { 
					grid.getPrototypeAt(p)[2] = r.nextDouble(); // set value
					break;
				}
				from = to;
			}
			
			double sumValue = 0;
			for( double[] d : grid.getPrototypes() )
				sumValue += d[2];
			
			if( sumValue > Math.sqrt(grid.size() ) * 10 )
				break;
		}
		
		// add dependent variable
		for( GridPos p : grid.getPositions() ) {
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( GridPos nb : grid.getNeighbours(p) )
				ds.addValue( grid.getPrototypeAt(nb)[2] );
			
			double[] pt = grid.getPrototypeAt(p);
			pt[3] = pt[2] + probSelect.get(p); // y Value
		}		
		return grid;
	}
	
	public static Grid2D<double[]> createSpDepGrid2(int xDim, int yDim, double pow, boolean toroid ) {

		Random r = new Random();
		Grid2D<double[]> grid;
		if( toroid )
			grid = new Grid2DToroid<double[]>(xDim, yDim);
		else
			grid = new Grid2D<double[]>(xDim, yDim);
					
		Map<GridPos,Double> probSelect = new HashMap<GridPos,Double>();
		List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
		Collections.shuffle(pos);
		
		for( GridPos p : pos ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			grid.setPrototypeAt(p, new double[]{i,j,0,0});
			probSelect.put(p,1.0);
		}
		
		while( true ) {		
			double sumProb = 0;
			for( double d : probSelect.values() )
				sumProb += d;
				
			double rnd = r.nextDouble()*sumProb;
			double from = 0, to;
			
			for( GridPos p : pos ) {
				to = from + probSelect.get(p);
				
				if( from <= rnd && rnd < to ) { 
					grid.getPrototypeAt(p)[2] = r.nextDouble(); // set value
					
					// update probabilities of p and all direct neighbors
					Set<GridPos> toUpdate = new HashSet<GridPos>();
					toUpdate.add(p);
					toUpdate.addAll(grid.getNeighbours(p));
					
					for( GridPos tp : toUpdate) {
						double pr = 1 + grid.getPrototypeAt(tp)[2];
						for( GridPos nb : grid.getNeighbours(tp) )
							pr += grid.getPrototypeAt(nb)[2];
						probSelect.put(tp,Math.pow(pr,pow));
					}	
					break;
				}
				from = to;
			}
			
			double sumValue = 0;
			for( double[] d : grid.getPrototypes() )
				sumValue += d[2];
			
			if( sumValue > Math.sqrt(grid.size() ) *10 )
				break;
		}
		
		//for( GridPos p : pos ) // real random
		//	grid.getPrototypeAt(p)[2] = r.nextDouble();
		
		// add dependent variable
		for( GridPos p : grid.getPositions() ) {
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( GridPos nb : grid.getNeighbours(p) )
				ds.addValue( grid.getPrototypeAt(nb)[2] );
			
			double[] pt = grid.getPrototypeAt(p);
			pt[3] = Math.pow(pt[2],2) + 0.1*ds.getMean(); // y Value
		}		
		return grid;
	}
	
	public static Grid2D<double[]> createSpDepGrid(int xDim, int yDim, double pow, boolean toroid ) {

		Random r = new Random();
		Grid2D<double[]> grid;
		if( toroid )
			grid = new Grid2DToroid<double[]>(xDim, yDim);
		else
			grid = new Grid2D<double[]>(xDim, yDim);
		
			List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
			Collections.shuffle(pos);
			
		for( GridPos p : pos ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			grid.setPrototypeAt(p, new double[]{ i, j,r.nextDouble(), 0 });
		}
		
		Map<GridPos,Double> values = new HashMap<GridPos,Double>();
		for( GridPos p : pos ) {
			DescriptiveStatistics ds = new DescriptiveStatistics();
			ds.addValue(grid.getPrototypeAt(p)[2]);
			for(GridPos nb : grid.getNeighbours(p) )
				ds.addValue(grid.getPrototypeAt(nb)[2]);
			values.put(p,ds.getMean());
		}
		
		for( GridPos p : pos ) {
			grid.getPrototypeAt(p)[2] = values.get(p); 
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for(GridPos nb : grid.getNeighbours(p) )
				ds.addValue(grid.getPrototypeAt(nb)[2]);
			grid.getPrototypeAt(p)[3] = grid.getPrototypeAt(p)[2] - 2*ds.getMean();
			//grid.getPrototypeAt(p)[3] = grid.getPrototypeAt(p)[2] * ds.getMean();
			//grid.getPrototypeAt(p)[3] = Math.pow(grid.getPrototypeAt(p)[2],2) + ds.getMean();
		}
			
		return grid;
	}
	
	public static List<Geometry> getGeoms(List<GridPos> positions ) {
		GeometryFactory gf = new GeometryFactory();
		List<Geometry> geoms = new ArrayList<Geometry>();
		for( GridPos p : positions ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			geoms.add( gf.createPolygon(new Coordinate[]{
					new Coordinate(i*50, j*50),
					new Coordinate((i+1)*50, j*50),
					new Coordinate((i+1)*50, (j+1)*50),
					new Coordinate(i*50, (j+1)*50),
					new Coordinate(i*50, j*50),
			}) );
		}
		return geoms;
	}
	
	public static void main(String[] args) {
		/*{
		Grid2D<double[]> grid = create1stOrderDoubleGrid(50,50,3,true);
		List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
		List<double[]> samples = new ArrayList<double[]>();
		for( GridPos p : pos )
			samples.add( grid.getPrototypeAt(p) );
		List<Geometry> geoms = getGeoms(pos);
		Drawer.geoDrawValues(geoms, samples, 2, null, ColorMode.Blues, "output/grid1st_x1.png");
		Drawer.geoDrawValues(geoms, samples, 3, null, ColorMode.Blues, "output/grid1st_y.png");
		}*/
		
		{
		Grid2D<double[]> grid = createSpDepGrid(50,50,3,true);
		List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
		List<double[]> samples = new ArrayList<double[]>();
		for( GridPos p : pos )
			samples.add( grid.getPrototypeAt(p) );
		List<Geometry> geoms = getGeoms(pos);
		Drawer.geoDrawValues(geoms, samples, 2, null, ColorMode.Blues, "output/grid2nd_x1.png");
		Drawer.geoDrawValues(geoms, samples, 3, null, ColorMode.Blues, "output/grid2nd_y.png");
		}
	}
}
