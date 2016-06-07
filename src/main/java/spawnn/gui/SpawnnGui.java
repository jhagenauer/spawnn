package spawnn.gui;

import java.awt.Dimension;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.miginfocom.swing.MigLayout;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.grid.GridPos;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class SpawnnGui extends JFrame implements PropertyChangeListener, ActionListener, ChangeListener, TrainingListener { //TODO A general class Prototype for GridPos, double[] etc. is needed

	private static final long serialVersionUID = -2728973134956990131L;

	private DataPanel dataPanel;
	private AnnPanel annPanel;

	private JTabbedPane tp;
	private JMenuItem aboutItem, quitItem;

	SpawnnGui() {
		super("SPAWNN");
				
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
		addWindowListener(new java.awt.event.WindowAdapter() {
		    @Override
		    public void windowClosing(java.awt.event.WindowEvent windowEvent) {
		        if (JOptionPane.showConfirmDialog(null,
		            "Are you sure to close this window?", "Really Closing?", 
		            JOptionPane.YES_NO_OPTION,
		            JOptionPane.QUESTION_MESSAGE) == JOptionPane.YES_OPTION){
		            System.exit(0);
		        }
		    }
		});
		

		Toolkit kit = Toolkit.getDefaultToolkit();
		Image[] icons = new Image[]{
				kit.createImage(ClassLoader.getSystemResource("icon_20x20.png")),
				kit.createImage(ClassLoader.getSystemResource("icon_26x26.png")),
				kit.createImage(ClassLoader.getSystemResource("icon_32x32.png")),
		};
		setIconImages(Arrays.asList(icons));

		// Menu Bar
		JMenuBar mb = new JMenuBar();
		mb.setLayout(new MigLayout("insets 0"));
		setJMenuBar(mb);

		JMenu fileMenu = new JMenu("File");
		
		quitItem = new JMenuItem("Quit");
		quitItem.addActionListener(this);
		fileMenu.add(quitItem);

		JMenu helpMenu = new JMenu("Help");
		aboutItem = new JMenuItem("About");
		aboutItem.addActionListener(this);
		helpMenu.add(aboutItem);

		mb.add(fileMenu, "push");
		mb.add(helpMenu);

		// Tabs
		tp = new JTabbedPane();
		dataPanel = new DataPanel(this);
		dataPanel.addPropertyChangeListener(this);

		tp.addTab("Data", dataPanel);

		annPanel = new AnnPanel(this);
		annPanel.addPropertyChangeListener(this);
		annPanel.addTrainingListener(this);

		tp.addTab("ANN", annPanel);
		tp.setEnabledAt(1, false);

		add(tp);
		tp.addChangeListener(this);

		pack();
		setSize(1245, 700);
		setMinimumSize(new Dimension(1245, 700)); // ugly, but needed because of ng results minimum size (vv)

		setVisible(true);
	}

	public static void main(String[] args) {
		/*try {
			UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
		} catch (Exception e) {
			try {
				UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}*/
		new SpawnnGui();
	}

	@Override
	public void propertyChange(PropertyChangeEvent e) {
		if (e.getPropertyName().equals(DataPanel.TRAIN_ALLOWED_PROP) )
			tp.setEnabledAt(1, (Boolean)e.getNewValue() );
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == aboutItem) {
			JOptionPane.showMessageDialog(this, "Spatial Analysis with (Self-Organinzing) Neural Networks (SPAWNN) ©2013-2015\n\n" + "All rights reserved.\n\n" + "Julian Hagenauer", "About", JOptionPane.INFORMATION_MESSAGE);
		} else if (e.getSource() == quitItem) {
			System.exit(1);
		} 
	}
	
	@Override
	public void stateChanged(ChangeEvent ce) {
		if( ((JTabbedPane)ce.getSource()).getSelectedComponent() == annPanel ) { // submit data for training
			int[] gaUsed = dataPanel.getGA(false);
			int[] gaAll = dataPanel.getGA(true);
			List<double[]> normedSamples = dataPanel.getNormedSamples();
			Map<double[],Map<double[],Double>> dMap = dataPanel.getDistanceMap();
			annPanel.setTrainingData( normedSamples, dataPanel.getSpatialData(), dataPanel.getFA(), gaUsed, gaAll, dMap);
		}
	}

	int numSom = 0;
	int numNg = 0;

	@Override
	public void trainingResultsAvailable(TrainingEvent te) {
		int[] fa = dataPanel.getFA();
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(dataPanel.getGA(false));
		List<double[]> samples = te.getSamples();
		SpatialDataFrame sdf = dataPanel.getSpatialData();
		int[] ga = dataPanel.getGA(true);
		
		if( te instanceof TrainingFinishedSOM ) { // SOM-results
			TrainingFinishedSOM tfs = (TrainingFinishedSOM)te;
			SOMResultPanel srp = new SOMResultPanel(this, sdf, samples, tfs.getBmus(), tfs.getGrid(), fDist, gDist, fa, ga, tfs.isWMC());
			tp.addTab("Results (SOM," + (numSom++) + ")", srp);
			tp.setSelectedComponent(srp);
			tp.setTabComponentAt(tp.indexOfComponent(srp), new ButtonTabComponent(tp));
			
			Map<double[],Set<double[]>> dBmus = new HashMap<>();
			for( Entry<GridPos,Set<double[]>> e :tfs.getBmus().entrySet() ) 
				dBmus.put( tfs.getGrid().getPrototypeAt(e.getKey()), e.getValue() );
			
			double a = DataUtils.getMeanQuantizationError(dBmus, fDist);
			double b = DataUtils.getMeanQuantizationError(dBmus, gDist);
			System.out.println("SOM "+(numSom-1));
			System.out.println("qe: "+a);
			System.out.println("sqe: "+b);
			System.out.println(""+a/b);
			
		} else {
			TrainingFinishedNG tfn = (TrainingFinishedNG)te;
			NGResultPanel srp = new NGResultPanel(this, sdf, samples, tfn.getBmus(), tfn.getGraph(), fDist, gDist, fa, ga, tfn.isWMC() );
			tp.addTab("Results (NG, " + (numNg++) + ")", srp);
			tp.setSelectedComponent(srp);
			tp.setTabComponentAt(tp.indexOfComponent(srp), new ButtonTabComponent(tp));
			
			double a = DataUtils.getMeanQuantizationError(tfn.getBmus(), fDist);
			double b = DataUtils.getMeanQuantizationError(tfn.getBmus(), gDist);
			System.out.println("NG "+(numNg-1));
			System.out.println("qe: "+a);
			System.out.println("sqe: "+b);
			System.out.println(""+a/b);
		}
	}
}
