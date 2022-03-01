Lab #1
Discovering the HPC Software Stack

Part 1: Discovering the environment

File topology.pdf contains the image of current node hardware topology.
This node has one socket (i.e., 1 CPU processor) with 6 physical cores (noted Core).
But there are 12 PUs, meaning that there are 12 logical cores (i.e., hyperthreads).
Regarding memory, the node contains 63GB of DRAM and a shared L3 cache of 12MB.
Each core has its own L1 cache (32KB for instructions and 32KB for data) and L2 cache 
(256KB for both instructions and data).
The network card is labeled "Ethernet".



Part 2: Running a job with Slurm

- Checking the cluster status

Running the 'sinfo' command returns the following output:

PARTITION   AVAIL  TIMELIMIT  NODES  STATE NODELIST
SallesInfo*    up 3-00:00:00    172   idle ablette,acromion,aerides,ain,albatros,allemagne,allier,anchois,angleterre,anguille,apophyse,ardennes,astragale,atlas,autriche,autruche,axis,barbeau,barbue,barlia,baudroie,belgique,bengali,bentley,brochet,bugatti,cadillac,calanthe,carmor,carrelet,charente,cher,chrysler,coccyx,corvette,cote,coucou,creuse,cubitus,cuboide,dindon,diuris,dordogne,doubs,encyclia,epervier,epipactis,espagne,essonne,faisan,femur,ferrari,fiat,finistere,finlande,ford,france,frontal,gardon,gelinotte,gennaria,gironde,groenland,gymnote,habenaria,harpie,hibou,hollande,hongrie,humerus,indre,ipsea,irlande,islande,isotria,jabiru,jaguar,jura,kamiche,labre,lada,landes,lieu,linotte,liparis,lituanie,loire,loriol,lotte,lycaste,malaxis,malleole,malte,manche,marne,maserati,mayenne,mazda,metacarpe,monaco,morbihan,moselle,mouette,mulet,murene,nandou,neotinea,nissan,niva,ombrette,oncidium,ophrys,orchis,parietal,perdrix,perone,peugeot,phalange,piranha,pleione,pogonia,pologne,pontiac,porsche,portugal,quetzal,quiscale,radius,raie,renault,requin,rolls,rotule,rouget,rouloul,roumanie,roussette,rover,royce,sacrum,saone,saumon,serapias,silure,simca,sitelle,skoda,sole,somme,sternum,suede,tarse,telipogon,temporal,test-[252-254],thon,tibia,traquet,truite,urabu,vanda,vanilla,vendee,venturi,verdier,volvo,vosges,xiphoide,xylobium,zeuxine

It contains 1 partition with 172 compute nodes. Everything is available (idle state)

- Submitting a job

Files test1.batch and test2.batch are provided to use SLURM.
