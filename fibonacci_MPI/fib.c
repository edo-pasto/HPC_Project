#include <stdio.h>
#include <mpi.h>

int main (int argc, char **argv){
  long n, fibn, x, y;
  int myrank, size;
  char command[] = "./fib";
  MPI_Comm children_comm[2];
  MPI_Comm parent;
  MPI_Info local_info;
  int world_size, universe_size, flag;
  int errcodes[1];
  
  MPI_Init (&argc, &argv);
  MPI_Comm_get_parent (&parent);
  MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
  MPI_Info_create (&local_info);

  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  if (parent == MPI_COMM_NULL)
    error ("No parent!");

  if (parent != MPI_COMM_NULL)
    MPI_Comm_remote_size (parent, &size);

  if (size != 1)
    error ("Something's wrong with the parent");

  MPI_Comm_get_attr (MPI_COMM_WORLD, MPI_UNIVERSE_SIZE, &universe_size, &flag);
 
  // if (!flag){
  //     printf("From fib.c This MPI does not support UNIVERSE_SIZE. How many\n processes total?");
      
  //     scanf ("%d", &universe_size);
  //     printf("From fib.c ");
      
  // }

  argv += 1;
  // printf("From fib.c ");
  n = atol (argv[0]);
  if (n < 2){
      printf ("<%ld> returning fib(n) < 2\n", n);

      MPI_Send (&n, 1, MPI_LONG, 0, 1, parent);

  }else{
      printf ("<%ld> spawning new process (1)\n", n);
      sprintf (argv[0], "%ld", (n - 1));

      MPI_Comm_spawn (command, argv, 1, local_info, myrank,
                      MPI_COMM_SELF, &children_comm[0], errcodes);

      printf ("<%ld> spawning new process (2)\n", n);
      sprintf (argv[0], "%ld", (n - 2));

      MPI_Comm_spawn (command, argv, 1, local_info, myrank,
                      MPI_COMM_SELF, &children_comm[1], errcodes);

      printf ("<%ld> waiting recv fib(n-1) > 2\n", n);

      MPI_Recv (&x, 1, MPI_LONG, MPI_ANY_SOURCE, 1,
                children_comm[0], MPI_STATUS_IGNORE);

        printf ("<%ld> waiting recv fib(n-2) > 2\n", n);

      MPI_Recv (&y, 1, MPI_LONG, MPI_ANY_SOURCE, 1,
                children_comm[1], MPI_STATUS_IGNORE);

      fibn = x + y;             // computation

      printf ("<%ld> returning fib(n) > 2\n", n);
      MPI_Send (&fibn, 1, MPI_LONG, 0, 1, parent);
    }
  printf ("<%ld> returned (isend) fib(n) \n", n);
  MPI_Finalize ();
}

