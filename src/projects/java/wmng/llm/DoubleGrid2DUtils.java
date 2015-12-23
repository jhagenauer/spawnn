package wmng.llm;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DToroid;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

public class DoubleGrid2DUtils {
	
	private static Logger log = Logger.getLogger(DoubleGrid2DUtils.class);
	
	public static Grid2D<double[]> createSpDepGrid(int xDim, int yDim, double pow, boolean toroid ) {

		double noise = 0.2;
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
			values.put(p,ds.getMean() + noise * r.nextDouble() );
		}
		
		for( GridPos p : pos ) {
			grid.getPrototypeAt(p)[2] = values.get(p); 
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for(GridPos nb : grid.getNeighbours(p) )
				ds.addValue(grid.getPrototypeAt(nb)[2]);
			grid.getPrototypeAt(p)[3] = Math.pow(grid.getPrototypeAt(p)[2],2) + Math.pow(ds.getMean(),2) + noise * r.nextDouble();
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
		DecimalFormat df = new DecimalFormat("000");
		
		for( int i = 0; i < 200; i++ ){
			log.debug(i);
			Grid2D<double[]> grid = createSpDepGrid(50,50,3,true);
			try {
				GZIPOutputStream gzos = new GZIPOutputStream(new FileOutputStream("output/grid_"+df.format(i)+".xml.gz"));
				SomUtils.saveGrid(grid,  gzos );
				gzos.finish();
				gzos.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			if( i == 0 ) {
				GeometryFactory gf = new GeometryFactory();
				List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
				List<double[]> samples = new ArrayList<double[]>();
				List<Geometry> geoms = new ArrayList<Geometry>();
				for( GridPos p : pos ) {
					samples.add( grid.getPrototypeAt(p) );
					
					int x = p.getPos(0);
					int y = p.getPos(1);
					int size = 50;
					geoms.add( gf.createPolygon( new Coordinate[]{
						new Coordinate(x*size,y*size),
						new Coordinate(x*size,(y+1)*size),
						new Coordinate((x+1)*size,(y+1)*size),
						new Coordinate((x+1)*size,y*size),
						new Coordinate(x*size,y*size),
					}));
				}
				DataUtils.writeShape(samples, geoms, new String[]{"i","j","x1","y"}, "output/grid.shp");
				
				Map<double[],Map<double[],Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(grid)));
				for( int j = 2; j <= 3; j++ ) {
					
					Map<double[],Double> values = new HashMap<double[],Double>();
					for( double[] d : samples )
						values.put(d,d[j]);
					double[] moran = GeoUtils.getMoransIStatisticsMonteCarlo(dMap, values, 999);
					log.debug("var "+j+", moran: "+moran[0]+", p-Value: "+moran[4]);
				}
			}
		}
	}
}
