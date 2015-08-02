package clustering_cng;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class CalcHomogenityOfPAS {

	public static void main(String[] args) {
		SpatialDataFrame sd1 = DataUtils.readShapedata( new File("data/philadelphia/planning/pas.shp"), new int[]{}, true);
		SpatialDataFrame sd2 = DataUtils.readShapedata( new File("output/data_philly.shp"), new int[]{}, true);
		
		for( int i = 0; i < sd2.samples.get(0).length; i++ ) { // for each attribute
			
			double sum2 = 0;
			
			for(Geometry g1 : sd1.geoms ) { // for each cluster
				
				List<double[]> l = new ArrayList<double[]>();
				for( int j = 0; j < sd2.samples.size(); j++ ) {
					Geometry g2 = sd2.geoms.get(j);
					if( g1.intersects(g2) )
						l.add(sd2.samples.get(j) );
				}
				
				// get mean
				double mean = 0;
				for( double[] d : l )
					mean += d[i];
				mean /= l.size();
				
				double sum = 0;
				for( double[] d : l ) {
					sum += Math.sqrt( Math.pow( mean - d[i], 2));
				}
				sum2 += sum/l.size();
			}
			
			System.out.println(sd2.names.get(i)+" : "+(sum2/sd1.geoms.size() ) );
			
			
		}

	}

}
