


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.BatchSOM;
import spawnn.som.utils.SomToolboxUtils;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class IrisTestBatch {

	private static Logger log = Logger.getLogger(IrisTestBatch.class);
	
	public static void main(String[] args) {
		int X_DIM = 7;
		int Y_DIM = 5;
						
		List<double[]> samples = null;
		try {
			
			samples = DataUtils.readCSV( new FileInputStream(new File("data/iris.csv") ) );			
			DataUtils.normalize(samples);
			
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		
		int EPOCHS = 100000/samples.size();
		log.debug("Epochs: "+EPOCHS);
		
		Dist<double[]> eDist = new EuclideanDist();
		
		int vLength = samples.get(0).length;
		Grid2DHex<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM );
		SomUtils.initLinear(grid, samples, true);
			
		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(eDist);
		
		long time = System.currentTimeMillis();
		BatchSOM som = new BatchSOM( new GaussKernel(grid.getMaxDist()), grid, bmuGetter );
		for (int t = 0; t < EPOCHS; t++) {
			som.train( (double)t/EPOCHS, samples );
		}
		log.debug("Took: "+(System.currentTimeMillis()-time)+"ms");
							
		log.debug("qe: "+SomUtils.getMeanQuantError( grid, bmuGetter, eDist, samples ) );
		log.debug("te: "+ SomUtils.getTopoError( grid, bmuGetter, samples ) );
		
		try {
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter );
			SomUtils.printDMatrix( grid, eDist, new FileOutputStream( "output/irisDmatrix.png" ) );
			SomUtils.printUMatrix( grid, eDist, new FileOutputStream( "output/irisUMatrix.png" ) );
			
			SomToolboxUtils.writeTemplateVector(samples, new FileOutputStream("output/iris.tv"));
			SomToolboxUtils.writeUnitDescriptions(grid, samples, bmus, eDist, new FileOutputStream("output/iris.unit"));
			SomToolboxUtils.writeWeightVectors(grid, new FileOutputStream("output/iris.wgt") );
			
			for( int i = 0; i < vLength; i++ )
				SomUtils.printComponentPlane(grid, i, new FileOutputStream("output/component"+i+".png"));
			
			SomUtils.saveGrid(grid, new FileOutputStream("output/grid.xml"));
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
