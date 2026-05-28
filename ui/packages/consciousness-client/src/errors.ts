export class GatewayError extends Error {
  readonly status: number;
  readonly reasonCode: string;
  readonly body: unknown;

  constructor(message: string, options: { status: number; reasonCode?: string; body?: unknown }) {
    super(message);
    this.name = "GatewayError";
    this.status = options.status;
    this.reasonCode = options.reasonCode ?? "gateway_error";
    this.body = options.body;
  }
}

export class ValidationError extends Error {
  readonly errors: string[];

  constructor(message: string, errors: string[]) {
    super(message);
    this.name = "ValidationError";
    this.errors = errors;
  }
}

export class SchemaVersionError extends Error {
  readonly expected: string;
  readonly received: string;

  constructor(expected: string, received: string) {
    super(`schema version mismatch: expected ${expected}, received ${received}`);
    this.name = "SchemaVersionError";
    this.expected = expected;
    this.received = received;
  }
}
